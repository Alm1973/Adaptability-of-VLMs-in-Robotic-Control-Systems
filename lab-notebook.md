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
