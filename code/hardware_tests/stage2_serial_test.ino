

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

#define SERVO_CHANNEL 0
#define SERVO_MIN 75
#define SERVO_MAX 580

int baseAngle = 90;
String cmd = "";

int angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setServo(int channel, int angle) {
  pca.setPWM(channel, 0, angleToPulse(angle));
}

void handle(String command) {
  command.trim();

  if (command == "ROTATE_BASE_LEFT_15") {
    baseAngle = constrain(baseAngle - 15, 0, 180);
  } else if (command == "ROTATE_BASE_RIGHT_15") {
    baseAngle = constrain(baseAngle + 15, 0, 180);
  } else if (command == "PING") {
    Serial.println("PONG");
    return;
  } else {
    Serial.println("UNKNOWN_COMMAND");
    return;
  }

  setServo(SERVO_CHANNEL, baseAngle);
  Serial.print("OK angle=");
  Serial.println(baseAngle);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pca.begin();
  pca.setPWMFreq(50);
  setServo(SERVO_CHANNEL, baseAngle);

  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handle(cmd);
      cmd = "";
    } else {
      cmd += c;
    }
  }
}
