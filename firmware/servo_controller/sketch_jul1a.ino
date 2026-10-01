#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define BASE 0
#define TILT 1

// Real bench-tested calibration per joint
#define BASE_MIN    75
#define BASE_CENTER 250
#define BASE_MAX    460

// ASSUMED to be Wrist/Camera tilt -- CONFIRM before flashing
#define TILT_MIN    140
#define TILT_CENTER 380
#define TILT_MAX    580

// ---------------------------------------------------------------------------
// SMOOTH MOTION (added 2026-08-12)
//
// Previously each serial command wrote the new angle straight to the PWM
// driver, so the servo jumped to it at its own maximum slew rate. Every
// correction -- even a single degree -- was a hard snap, and a 12-degree
// search step was a violent swing. That is the main source of the jerkiness,
// and no amount of host-side smoothing can fix it: by the time the host sends
// a command the firmware has already committed to moving as fast as it can.
//
// So TARGET and CURRENT angle are now separate. Serial commands move the
// TARGET instantly (host-side bookkeeping in arm.py is unchanged and still
// correct -- self.base is the target). A fixed-rate loop eases CURRENT toward
// it and drives the PWM from that.
//
// The easing is proportional (step = distance * EASE) with a hard speed cap:
//   * the cap bounds top speed, so large moves are never violent
//   * the proportional term decelerates on approach, so the arm ARRIVES
//     softly instead of stopping dead -- which is what actually reads as
//     "smooth" rather than merely "slow"
// Sub-degree steps matter here, hence the float pulse mapping below: at
// ~2.1 pulse counts per degree on the base, a 0.45 deg step is ~1 count, so
// motion advances one PWM count at a time.
//
// Timing: MAX_DEG_PER_STEP / STEP_MS = 0.45 deg per 12 ms = ~37 deg/sec, so a
// 12-degree search step completes in roughly 0.4 s -- comfortably inside
// find_live's SEARCH_COOLDOWN and live_experiment's MOVE_PAUSE of 0.7 s.
// ---------------------------------------------------------------------------
const unsigned long STEP_MS = 12;      // easing tick period
const float MAX_DEG_PER_STEP = 0.45;   // speed cap -> ~37 deg/sec
const float EASE = 0.25;               // proportional approach (soft arrival)
const float SNAP_DEG = 0.05;           // close enough: land exactly on target

int   baseTarget = 90, tiltTarget = 90;   // where we are going
float baseCur    = 90.0, tiltCur = 90.0;  // where we are now
int   basePulse  = -1, tiltPulse = -1;    // last pulse written, -1 = never
unsigned long lastStep = 0;

// Non-blocking line assembly. Serial.readStringUntil('\n') blocks for up to
// the serial timeout when a line arrives in pieces, which would stall the
// easing loop mid-move and produce exactly the stutter this rewrite removes.
char lineBuf[32];
byte lineLen = 0;

int clampInt(int v, int minv, int maxv) {
  if (v < minv) return minv;
  if (v > maxv) return maxv;
  return v;
}

// Piecewise linear using the real calibrated center as the breakpoint,
// so it respects the asymmetric range instead of assuming a simple
// symmetric 0-180 -> min-max stretch.
// Float input: the easing works in fractions of a degree, and rounding to a
// whole degree first would quantise the ramp back into visible steps.
int angleToPulse(float angle, int minPulse, int centerPulse, int maxPulse) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  if (angle <= 90) {
    return (int)(minPulse + (centerPulse - minPulse) * (angle / 90.0) + 0.5);
  } else {
    return (int)(centerPulse + (maxPulse - centerPulse) * ((angle - 90.0) / 90.0) + 0.5);
  }
}

void applyServo(int channel, float angle, int minPulse, int centerPulse,
                int maxPulse, int *lastPulse) {
  int pulse = angleToPulse(angle, minPulse, centerPulse, maxPulse);
  // Only touch I2C when the pulse actually changes. During the slow tail of
  // an ease many ticks map to the same count, and re-sending it is pure bus
  // traffic.
  if (pulse != *lastPulse) {
    pwm.setPWM(channel, 0, pulse);
    *lastPulse = pulse;
  }
}

void stepAxis(int channel, int target, float *cur, int minPulse,
              int centerPulse, int maxPulse, int *lastPulse) {
  float d = (float)target - *cur;
  if (d < SNAP_DEG && d > -SNAP_DEG) {
    if (*cur != (float)target) {
      *cur = (float)target;
      applyServo(channel, *cur, minPulse, centerPulse, maxPulse, lastPulse);
    }
    return;
  }
  float step = d * EASE;
  if (step >  MAX_DEG_PER_STEP) step =  MAX_DEG_PER_STEP;
  if (step < -MAX_DEG_PER_STEP) step = -MAX_DEG_PER_STEP;
  *cur += step;
  applyServo(channel, *cur, minPulse, centerPulse, maxPulse, lastPulse);
}

void handleLine(char *s) {
  char *xp = strstr(s, "X:");
  char *yp = strstr(s, "Y:");
  if (xp == NULL || yp == NULL) return;

  int xVal = atoi(xp + 2);
  int yVal = atoi(yp + 2);

  // SIGN FIX 2026-08-11. This was `-=`, while arm.py tracked `+`, so the
  // two disagreed: after 17x "X:+1" arm.py believed base=107 while the
  // firmware sat at 73. They mirrored around 90. Harmless for clamping
  // (both hit limits together) and for re-home (self-consistent), but
  // FATAL for anything reading arm.base semantically -- main.py's
  // REACQUIRE would infer "panned right" when physically panned left.
  // arm.py now negates to match, so PHYSICAL DIRECTION IS UNCHANGED and
  // arm.base finally equals the true servo angle.
  //
  // Note this now moves the TARGET, not the servo. Clamping stays here so
  // the target can never wander outside [0,180] and accumulate an offset the
  // host does not know about -- arm.py clamps identically, so the two agree.
  baseTarget = clampInt(baseTarget + xVal, 0, 180);
  tiltTarget = clampInt(tiltTarget + yVal, 0, 180);
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      // Overlong garbage: drop the whole line rather than truncating it into
      // a command that happens to parse.
      lineLen = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(1000);

  // Explicitly move both servos to their real calibrated center on boot,
  // matching the code's own starting assumption of angle 90/90.
  // Deliberately NOT eased: on power-up the true physical position is
  // unknown, so there is nothing meaningful to ease FROM. home.py relies on
  // this happening on port open.
  applyServo(BASE, baseCur, BASE_MIN, BASE_CENTER, BASE_MAX, &basePulse);
  applyServo(TILT, tiltCur, TILT_MIN, TILT_CENTER, TILT_MAX, &tiltPulse);
  lastStep = millis();
}

void loop() {
  readSerial();

  unsigned long now = millis();
  if (now - lastStep >= STEP_MS) {
    lastStep = now;
    stepAxis(BASE, baseTarget, &baseCur, BASE_MIN, BASE_CENTER, BASE_MAX,
             &basePulse);
    stepAxis(TILT, tiltTarget, &tiltCur, TILT_MIN, TILT_CENTER, TILT_MAX,
             &tiltPulse);
  }
}
