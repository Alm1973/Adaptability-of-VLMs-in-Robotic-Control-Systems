# Hardware

Photos of the assembled AVI hardware — 4-servo arm with camera mount, and the Arduino/PCA9685/UBEC power wiring.

## Camera + servo arm

![Camera and servo arm, angle 1](images/hardware-1.jpg)

![Camera and servo arm, angle 2](images/hardware-2.jpg)

Logitech C270 webcam mounted at the top of the arm, driven by a stack of DS3218 servos (base, shoulder, elbow, wrist). See [`project-database.md`](project-database.md#31-servo-calibration-final-values) for per-servo PWM calibration values.

## Control + power wiring

![Arduino, PCA9685, UBEC, and battery wiring](images/hardware-3.jpg)

Arduino Uno (USB-connected to laptop) driving a PCA9685 servo driver over I2C, powered by a UBEC off a LiPo battery. Servo power and Arduino ground are tied together — see [`system-schematic.md`](system-schematic.md#physical-wiring-notes) for why this matters.
