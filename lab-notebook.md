VLM Adaptability Project


## 2026-05-26 — Latency Optimization + GPU Acceleration (MLX)

improving inference latencies for the vision-language pipeline. I enabled GPU computation via MLX which improved performance relative to CPU powered LLaVA.  shift from extremely long inference periods to a relatively shorter ones. One thing that was effective was using the GPU for computation which greatly lessened computational bottlenecks in reasoning processes.


## 2026-05-30 — Hardware Loop + Camera Integration (OpenCV)

Tried to link the vision system to a live camera feed via OpenCV and incorporate the same with the hardware loop of the robot. Got errors in initializing the camera and other backend errors like AVFoundation and Device Index error. The vision system failed to gain access to the camera feed thereby interrupting the loop. That which worked was the OpenCV pipeline itself along with a portion of the algorithm for detecting the object when frames are manually supplied. That which did not work was the failure to access the camera along with the synchronization of the hardware loop.



## 2026-06-05 — System Integration Brainstorming (Robot Behavior Design)

Proposed some concepts at the system level regarding the integration of perception, reasoning, and actuation through the robot and the overall design of thr robot. 


## 2026-06-10 — README + Lab Notebook Setup (Project Organization)

Repository setup and creation of the initial documentation process for the Robot Vision Project. The direction of the project determined by the need to create adaptable vision-language models for robotic applications. Starting to create organized experiments in a lab notebook for iterations. The right thing was creating documentation and organizing experiments with links to system’s purpose. Wrong way to do it was the previous lack of tracking the experiments since there wasn’t any point of comparison between iterations. The next step will be to document all hardware and model changes and start benchmarks.

---

## 2026-06-14 — Hardware Loop Closed (PCA9685 + 4 Servos)

Got Python ↔ Arduino serial communication fully working through the PCA9685 16-channel PWM driver, with all four servos responding to commands. Calibrated a HOME position and a smooth REST animation, and confirmed the serial connection is stable across repeated commands. This closes the "Python → Arduino → Servos" stage of the pipeline; next step is connecting the camera stage (OpenCV frame capture) and then the VLM reasoning stage on top of it, moving toward the full perception → decision → action loop.

## 2026-06-22 — Per-Servo Calibration Finalized

Replaced the original global PWM mapping (`map(angle, 0, 180, 75, 580)`) with individual calibrated MIN/CENTER/MAX values for each servo, since each joint has a different mechanical range. Final values: Servo 0 (base) 75/250/460, Servo 1 (shoulder/tilt) 75/370/580, Servo 3 (wrist) 80/430/570, Servo 4 140/380/580. What worked was isolating each servo's safe range individually instead of assuming one range fits all joints; what didn't work was the original global assumption, which risked jitter/stall at the edges for some servos. Next step is wrapping this into a single `setServoAngle(servoID, angle)` helper so raw PWM values never need to be touched directly again.

## 2026-06-29 — Modular AVI System + Red Cup Tracking

Refactored the project into a proper modular package (`main.py`, `planner.py`, `vision.py`, `controller.py`, `state_manager.py`, `colors.py`, plus `avi_control.ino`) instead of one-off scripts, and got autonomous red-cup tracking working end-to-end: HSV-based detection → centroid error from frame center → proportional base/wrist servo correction → live state reporting (SEARCHING / TRACKING / REACQUIRE) in the terminal. Debugged a stuck-in-REACQUIRE failure mode caused by two issues: no guaranteed transition back to TRACKING on a fresh detection, and the "scan" recovery command not actually driving the servo. Also caught and fixed a camera index mismatch (index 0 vs. 1) that was silently producing black frames. Next step is tuning scan sweep behavior and VLM polling interval so recovery feels responsive rather than sluggish.

## 2026-07-02 — VLA/π0 Evaluation + Hybrid Architecture Confirmed

Investigated whether switching to a Vision-Language-Action model (π0 / OpenVLA) would be a more efficient replacement for the current OpenCV + Qwen2.5-VL pipeline. Concluded no: π0-class models assume 6-7 DOF arms with force/joint feedback and manipulation tasks, which doesn't match AVI's 4-servo, camera-only search task, and swapping to an end-to-end policy would change the research question away from studying explicit reasoning/recovery. Confirmed the hybrid architecture instead — OpenCV handles deterministic perception (centroid tracking via PID when tracking is healthy), Qwen2.5-VL only gets invoked for higher-level recovery reasoning when tracking is lost — and mapped this onto a 4-layer stack (perception / state tracker / policy / actuation) so future upgrades (YOLO, RL, VLA) can swap into individual layers without rewriting the whole system. Also paused Isaac Sim simulation work due to tooling friction (URDF/USD joint path discovery, version mismatches) in favor of continued iteration on the real hardware loop. Next step: build a clean "look-at-object" loop as the stable baseline before layering in richer adaptive reasoning (memory of past viewpoints, failure-history-conditioned search).

---
