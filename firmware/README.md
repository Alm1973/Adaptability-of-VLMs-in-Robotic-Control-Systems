# Firmware

`servo_controller/sketch_jul1a.ino` runs on the Arduino Uno and drives the servos through a PCA9685 PWM board over I2C (Adafruit PWM Servo Driver library).

- Serial runs at 115200 baud. Each line is a relative move, `X:<degrees> Y:<degrees>` (for example `X:+2 Y:-1`). `X` moves the base (channel 0) and `Y` moves channel 1.
- Channel 1 is the shoulder. The sketch calls it `TILT`, but a hardware check on 1 September 2026 showed it swings the whole arm. See `docs/servo-calibration.md`.
- Commands move a target angle, and a 12 ms loop eases the servo toward it at up to about 37°/s.
- Both angles are clamped to 0° to 180°, then mapped to the pulse ranges set in the sketch. Channel 1 uses the wrist servo's range (140/380/580) rather than the shoulder's own (75/370/580); `docs/servo-calibration.md` has the details.
- Opening the serial port resets the Uno, which moves both servos straight to 90°/90°.

Servo power comes from a separate UBEC. Its ground must be connected to the Arduino's ground.
