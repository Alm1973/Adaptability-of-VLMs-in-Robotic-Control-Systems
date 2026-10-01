#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;

const int s1Pin = 3;
const int s2Pin = 5;
const int s3Pin = 6;
const int s4Pin = 9;

int pos1 = 90;
int pos2 = 90;
int pos3 = 90;
int pos4 = 90;


void moveSlow(Servo &servo, int &currentPos, int targetPos, int stepDelay) {

  int step = (currentPos < targetPos) ? 1 : -1;

  while (currentPos != targetPos) {
    currentPos += step;
    servo.write(currentPos);
    delay(stepDelay);
  }
}

void setup() {
  servo1.attach(s1Pin);
  servo2.attach(s2Pin);
  servo3.attach(s3Pin);
  servo4.attach(s4Pin);

  servo1.write(90);
  servo2.write(90);
  servo3.write(90);
  servo4.write(90);

  delay(1500);
}

void loop() {

  

  moveSlow(servo1, pos1, 30, 25);
  moveSlow(servo2, pos2, 120, 25);
  moveSlow(servo3, pos3, 60, 25);
  moveSlow(servo4, pos4, 45, 25);
  delay(2000);

  moveSlow(servo1, pos1, 90, 25);
  moveSlow(servo2, pos2, 60, 25);
  moveSlow(servo3, pos3, 120, 25);
  moveSlow(servo4, pos4, 135, 25);
  delay(2000);

  moveSlow(servo1, pos1, 150, 25);
  moveSlow(servo2, pos2, 90, 25);
  moveSlow(servo3, pos3, 30, 25);
  moveSlow(servo4, pos4, 90, 25);
  delay(2000);

  moveSlow(servo1, pos1, 90, 25);
  moveSlow(servo2, pos2, 90, 25);
  moveSlow(servo3, pos3, 90, 25);
  moveSlow(servo4, pos4, 90, 25);

  delay(600000); // 10 minute pause
}