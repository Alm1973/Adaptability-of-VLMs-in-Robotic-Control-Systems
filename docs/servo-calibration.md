# Servo Calibration Reference

Raw PWM calibration values for the arm's 4 servos, as provided by the
user on 2026-08-04. Reference only — not yet wired into any code.

| Servo | Joint | Min PWM | Center PWM | Max PWM |
|---|---|---|---|---|
| 0 | Base rotation | 75 | 250 | 460 |
| 1 | Shoulder (up/down) | 75 | 370 | 580 |
| 2 | Elbow | 80 | 430 | 570 |
| 3 | Wrist / Camera tilt | 140 | 380 | 580 |

## RESOLVED 2026-09-01 — channel 1 is the SHOULDER

Confirmed by observation with the operator watching (`hw_check.py`): commanding
`Y:` moves the **shoulder**, not the wrist. So the channel numbering in the
table above is CORRECT, and the firmware's names are wrong.

`sketch_jul1a.ino` calls channel 1 `TILT` and drives it with **servo 3's**
pulse range (140/380/580) rather than the shoulder's own (75/370/580). It
under-travels at the low end by ~30% of the range. Safe, but wrong, and
nothing in the code explains the missing travel.

Direction, also confirmed: `arm.update(+10, 0)` swings the view RIGHT;
`arm.update(0, +10)` moves the view DOWN.

**Consequence worth carrying into the writeup:** the second axis TRANSLATES
the camera rather than rotating it, because the shoulder moves the whole arm.
The project's ego-motion reasoning (`_periphery_diff`, the DISPLACED state)
was empirically calibrated on this hardware so its thresholds hold, but it was
designed against a pan/tilt mental model that does not match the machine.

Servos 2 (elbow) and 3 (wrist) are driven by nothing and hold wherever left.

## Known gap with the current codebase

`arm.py`'s `Arm` class currently models only **2** axes — `self.base` and
`self.tilt` — as plain degrees (0-180), sent over serial as `X:{deg}
Y:{deg}\n`. It has no concept of PWM values, and no control over
shoulder or elbow (servos 1 and 2 above) at all.

Unconfirmed/unresolved as of this note:
- Whether `arm.py`'s `base`/`tilt` correspond to servo 0 (base rotation)
  and servo 3 (wrist/camera tilt) specifically, or something else.
- Whether shoulder/elbow (servos 1-2) are fixed/manually positioned
  hardware, unused, or controlled by firmware outside this repo.
- How the 0-180 degree values `arm.py` sends map to these PWM ranges --
  note the center PWM isn't the midpoint of min/max for any of the 4
  servos (e.g. base: 75/250/460 -- center is much closer to min than
  max), so a simple linear degree-to-PWM formula can't be assumed without
  confirmation.

Do not assume a mapping and write code against it without checking back
first -- this is exactly the kind of assumption mismatch that caused the
prior arm-sweep incident (see `PROGRESS.md`).
