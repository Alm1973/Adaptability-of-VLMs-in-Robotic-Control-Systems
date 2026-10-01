

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

#define SERVO_CHANNEL 0
#define SERVO_MIN 75
#define SERVO_MAX 580

int angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setServo(int channel, int angle) {
  pca.setPWM(channel, 0, angleToPulse(angle));
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pca.begin();
  pca.setPWMFreq(50);

  Serial.println("Stage 1 test: sweeping servo 0 to confirm PCA9685 wiring.");
}

void loop() {
  // Sweep 0 -> 180 -> 0 slowly so you can visually confirm smooth motion.
  for (int angle = 0; angle <= 180; angle += 5) {
    setServo(SERVO_CHANNEL, angle);
    Serial.print("Angle: ");
    Serial.println(angle);
    delay(50);
  }
  delay(500);
  for (int angle = 180; angle >= 0; angle -= 5) {
    setServo(SERVO_CHANNEL, angle);
    Serial.print("Angle: ");
    Serial.println(angle);
    delay(50);
  }
  delay(1000);
}

/*
 * PASS CRITERIA (log this in lab-notebook.md):
 *   - Servo moves smoothly through the full sweep, no jitter or stall.
 *   - Serial monitor angle values match what you visually observe.
 * If it jitters or stalls near an edge, narrow SERVO_MIN/SERVO_MAX and
 * re-test before moving to Stage 2.
 */
