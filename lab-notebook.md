VLM Adaptability Project


## 2026-05-26 — Latency Optimization + GPU Acceleration (MLX)

Latency reduction of inference in the vision-language model. By allowing the use of GPUs through MLX, I have been able to improve latency in comparison to the CPU based version of LLaVA. From very long inference latencies to relatively short latencies. One solution that worked well was the use of the GPU in computations.


## 2026-05-30 — Hardware Loop + Camera Integration (OpenCV)

Attempted to connect the vision system to the live stream from a camera using OpenCV, and to include the same in the hardware loop of the robot. There were issues with initializing the camera and some backend issues such as AVFoundation and device index error. The vision system was unable to access the stream from the camera, which resulted in the loop being interrupted. What worked was the OpenCV pipeline as well as the object detection algorithm when frames were manually fed into the system. What did not work was accessing the camera and synchronizing the hardware loop.



## 2026-06-05 — System Integration Brainstorming (Robot Behavior Design)

Proposed some concepts at the system level regarding the integration of perception, reasoning, and actuation through the robot and the overall design of thr robot. 


## 2026-06-10 — README + Lab Notebook Setup (Project Organization)

Setup of the repository and creation of the process of documenting the initial phase of the Robot Vision Project. Direction of the project chosen according to the necessity of developing adaptive vision-language models for use in robotics. Start of creating well-organized experiments in the lab notebook for iterations. The correct way to approach this was documenting and organizing the experiments, linking to the purpose of the system. The incorrect way to do this is the previous lack of recording of the experiments because there was no point of comparison between iterations.

---

## 2026-06-14 — Hardware Loop Closed (PCA9685 + 4 Servos)

Successfully got Python ↔ Arduino serial communication running completely with the help of PCA9685 16-channel PWM driver for all four servos. Got a calibration for both a HOME position and REST animation, and made sure that the serial communication is stable and working fine on repeat. Thus, we can consider the "Python → Arduino → Servos" phase completed, and now we need to integrate the "Camera" (OpenCV frame capturing) and "VLM reasoning" phase above it.

## 2026-06-22 — Per-Servo Calibration Finalized

Global PWM mapping (`map(angle, 0, 180, 75, 580)`) replaced by separate calibration for each individual servo’s MIN / CENTER / MAX because different mechanical ranges for each joint. Finally got: Servo 0 (base): 75 / 250 / 460, Servo 1 (shoulder/tilt): 75 / 370 / 580, Servo 3 (wrist): 80 / 430 / 570, Servo 4: 140 / 380 / 580. The successful approach was the individual safe range for each servo separately instead of making an assumption that one range will work for all the joints; the unsuccessful approach was the original global assumption, which could lead to jittering/stalling of some servos at extreme values.

## 2026-06-29 — Modular AVI System + Red Cup Tracking

Converted the project into a real modular package, which includes `main.py`, `planner.py`, `vision.py`, `controller.py`, `state_manager.py`, `colors.py`, along with `avi_control.ino`. Autonomous tracking of red cups is now achieved from end-to-end: HSV detection -> Error calculation using centroid and frame center -> Base/Wrist servo adjustment using proportional control -> Reporting states (SEARCHING / TRACKING / REACQUIRE) to terminal. There were bugs that led to the system being stuck in the state of REACQUIRE, because of: no guarantee that the system will go to TRACKING state after a new detection, and no execution of the "scan" command, which should rotate the servo. Another bug that was found and fixed was an index conflict with the camera, causing black frames to be displayed.

## 2026-07-02 — VLA/π0 Evaluation + Hybrid Architecture Confirmed

Probed whether a switch to Vision-Language-Action (π0/OpenVLA) would be an efficient replacement for the existing OpenCV+Qwen2.5-VL setup. Decided not to as π0-models assume 6-7 DOF robots with force/joint feedback and manipulation, rather than 4-servo AVI with only visual feedback and a search task. The switch would also shift the research goal from studying reasoning/recovery to studying end-to-end control and hence decided not to. Rather went ahead with the hybrid architecture wherein OpenCV takes care of deterministic perception (tracking of centroid using PID in healthy tracking state), while Qwen2.5-VL gets called into action only for higher level reasoning when tracking fails. Formulated a neat 4-layer framework of (perception -> state tracker -> policy -> actuation) where different parts can be upgraded separately in the future (YOLO, RL, VLA) as each can plug in directly as an individual layer in the stack. In addition, decided to pause the Isaac Sim simulation efforts due to tooling issues (URDF/USD joint path discovery, version incompatibility). Next steps: create a "clean look-at-object" loop with AVI as the stable benchmark to start from before adaptive reasoning layers get added on.

## Retroactive Entry — Latency Benchmark (Assignment 7b)

Experimented with five different configurations for the reasoning part of the pipeline in search of something that could be used for real-time robotic control. The CPU-based baseline for LLaVA was at ~67.8s per inference and thus not viable due to extreme latencies. Enabling the use of the GPU via MLX reduced it to ~12.0s, which then came out as ~6s after another run, confirming that the bottleneck here was computational and that some latency could potentially be fixed by offloading computations to the GPU. Using the lighter model (Moondream) rather than a heavier one proved to be the single most effective step toward reducing the latency, reducing it to ~4.8s — the only local solution that approached real-time viability. The call to cloud APIs turned out to be the quickest of all, but they were ruled out as a primary option due to introducing the network latency that should have been avoided.

## Retroactive Entry — Spatial Reasoning Progression (Assignment 7b)

Accuracy was tracked using three iterations of prompts/architecture on the centering task. An unstructured prompt resulted in poor accuracy and inconsistent results. Structuring the prompt with explicit spatiality (defined pixel regions, defined directional relations) doubled the accuracy of the model compared to the unstructured prompt, reinforcing the hypothesis that explicitly providing the model with the coordinate system is more effective than previously thought. The final hybrid approach of having OpenCV calculate the true centroid/grid location and only providing the VLM with the label of that location instead of trying to reason out where it was located from raw pixels achieved 89% accuracy on the centering task (from approximately 33% in early experiments), with anchor-point prompting also boosting cycle-to-cycle consistency. This directly motivates the chosen architecture: OpenCV localization, VLM reasoning. Next step: assignment 2's extended grid test to verify the 89% number in all nine possible grid locations.

## Retroactive Entry — Servo Driver Failure

The PCA9685 servo driver used previously for the project was defective (defective is the term used in the mentor communication) and had to be replaced before the hardware loop could be completed, which is why the first task in the hardware-loop assignment involves ordering the exact same part again. The replacement PCA9685 boards are inexpensive ($6-$10) and easily obtainable, so the repair is simple when the replacement comes in; the more important point to take from this is that this is exactly the type of single point of failure that should be caught by the hardware tests done in stages in `code/hardware_tests/` (Stage 1 tests the servo driver alone without any other components connected to it, thus catching a problem right away and not after the loop has been put together).

---
