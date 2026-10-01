#  Project Database


## 1. Research Question & Framing

**Working research question:**
> Can a locally run VLM autonomously recover from task-disrupting situations occlusion, environmental change, and abnormal/anomalous targets through dynamic reasoning and adaptive camera repositioning, during an object verification task?


-  **occlusion**, **environmental change**, **anomaly** .

**Metrics identified for evaluating success:**
- Recovery time (seconds to adapt)
- Task success rate (%)
- Number of retries
- Error type classification (perception, planning, actuation)


## 2. System Architecture

### 2.1 Architecture decision (Architecture A vs. B)

**Architecture A — "Pixy2 sees, LLM thinks"**
`Pixy2 (onboard color-signature detection) → Arduino → Raspberry Pi → small text LLM (Qwen 0.5B–3B, Ollama)`
- LLM never sees an image — only structured text like `Object detected: signature=2 (red), x=152, y=88...`
- Pros: uses existing hardware, lower compute.
- Cons: Pixy2 limited to color blobs can't represent shape or detail which weakens the "anomaly" failure category.

**Architecture B — "VLM sees and thinks"**
`Camera (webcam/Pi Camera) → VLM (image input) → reasoning + action, on laptop or cloud API`
- Pros: rich perception, natural fit for anomaly detection, no Pi needed.
- Cons: doesn't run at usable speed on a Raspberry Pi 5 needs laptop or cloud compute.

**Decision:** The project moved forward with a **VLM-based approach** (Architecture B direction), ultimately running **Qwen2.5-VL:3B via Ollama** on the device locally instead of using cloud API, using a USB webcam (Logitech C270, 720p) and Arduino to actuate. 

### 2.2 Final hybrid pipeline

The system converged on a **hybrid OpenCV + VLM architecture** rather than pure VLM-only or pure classical CV:

```
OpenCV (perception / object detection)
        │
        ▼
   Tracking OK?
   ┌────┴────┐
  Yes         No
   │           │
  PID      Qwen2.5-VL (high-level reasoning / recovery strategy)
   │           │
   └─────┬─────┘
         ▼
   State Machine
         ▼
     Arduino → Servos
```


Planned phased rollout:
- **Phase 1:** OpenCV → object centroid → move arm to center (baseline, deterministic).
- **Phase 2:** Replace/augment OpenCV with YOLO or VLM perception.
- **Phase 3:** Replace rule-based policy with RL or VLA.

### 2.3 VLA / π0 evaluation (rejected for now)

Potential alternatives to the entire pipeline, π0 (Physical Intelligence) and OpenVLA, were considered. Conclusion: **not used**, due to the following reasons:
- π0 models assume robots with 6-7 DOFs with force sensing, joint encoders and gripper manipulations – AVI has 4 servos without encoders, force sensors, only a visual task (search, no manipulations)
- Switching to VLA from the reasoning loop will remove what is being studied, i.e. explicit multi-step reasoning/recovery, and make the process "see → do".
- Suggested direction: enhance reasoning in AVI by adding viewpoint memory, object hypothesis about its possible location, choice of search strategy depending on the scene conditions.

### 2.4 Simulation (Isaac Sim) — parked

Isaac Sim/URDF import setup for Panda Arm simulation environment was tried but experienced significant tooling friction (version incompatibilities, no user interface tools, USD joint path discovery). Decision made: **paused**, in favor of using lighter OpenCV + VLM loop on real hardware.

---

## 3. Hardware

- Raspberry Pi 5 (tested in Architecture A; currently the trend is moving towards Qwen running on-device as per the latest state of the project)
- Arduino (Mega, eventually with a PCA9685 16 channels PWM controller for servos)
- 4× DS3218 20kg·cm digital metal-gear servos (270°, waterproof) for base / shoulder / elbow / wrist (camera pan tilt alternative)
- UBEC for servos power (UBEC must be of the capacity of ≥10A, better if 15A, considering the peak of up to 9A from 3× 20kg servos)
- Logitech C270 webcam 720p USB (lightweight mountable and good enough resolution for 25-35cm height and 150x150cm workspace)
- Ground connection between UBEC and Arduino is an important aspect of wiring (most common point of failure)
### 3.1 Servo calibration (final values)

Per-servo calibration replaced an earlier single global PWM mapping (`map(angle, 0, 180, 75, 580)`) 

| Servo | Role | MIN (PWM) | CENTER (PWM) | MAX (PWM) |
|---|---|---|---|---|
| Servo 0 | Base rotation | 75 | 250 | 460 |
| Servo 1 | Shoulder / tilt | 75 (front down) | 370 (up) | 580 (back down) |
| Servo 3 | Wrist | 80 | 430 | 570 |
| Servo 4 | (4th joint) | 140 | 380 | 580 |

Home position (early setup): `base=55, shoulder=115, elbow=125, wrist=110`.

---

## 4. Software / Codebase

Python package (`avi/`) plus Arduino :

```
avi/
 ├── main.py            # entry point, camera loop, state machine orchestration
 ├── planner.py          # recovery / search strategy planning
 ├── vision.py            # OpenCV detection (HSV masking, centroid)
 ├── controller.py        # serial communication to Arduino
 ├── state_manager.py     # SEARCHING / TRACKING / REACQUIRE state machine
 ├── colors.py             # HSV color range definitions
 └── avi_control.ino       # Arduino firmware (PCA9685 servo driver)
```

**Run sequence:**
1. Flash `avi_control.ino` to the Arduino (PCA9685-based).
2. `source ~/robotenv/bin/activate`
3. `pip install opencv-python numpy pyserial ollama` (once)
4. `ollama serve` and `ollama pull qwen2.5vl:3b`
5. `python main.py`

**State machine behavior:** `SEARCHING → TRACKING → REACQUIRE`, using scanning during searching/reacquiring and color tracking while the target is being tracked. Fixed problems include the inability of the system to get out of `REACQUIRE` mode if there was no guarantee of exit from it back to `TRACKING` after a new detection was made, and the absence of a real implementation for the "scan" movement command by Arduino – all of which have been fixed (increasing the missed frame limit before REACQUIRE and adding REACQUIRE → TRACKING transition on detection, replacing fake scanning with movements).

### 4.1 Object tracking (red-cup baseline)

Baseline tracking loop (`vision.py`-equiv): HSV dual-range red mask (`0-10` and `170-180` hue bands) → largest contour above an area threshold → bounding-box centroid → error from frame center → proportional base/wrist servo nudge (deadzone ~40px, send interval ~80ms) → serial write of `BASE:<angle>` / `WRIST:<angle>` commands to Arduino, which parses and drives the PCA9685.

### 4.2 Latency benchmarking results

Latency by approach:

| Approach | Latency | Note |
|---|---|---|
| CPU-based LLaVA (baseline) | ~67.8s | Too slow for real-time robotics |
| GPU-accelerated via MLX | ~12.0s (later reported ~6s) | Major improvement, still not fully real-time |
| Moondream (lightweight local model) | ~4.8s | Fastest local option tested |
| Llama.cpp runtime | faster than Ollama | Lower-level runtime reduces overhead |
| Cloud API | fastest overall | Adds network dependency + minor per-call cost |



**Timeout calibration methodology:** Full loop cycle = observation → decision → action → reobservation. Recommended value for timeout = 3–5× *the worst possible* cycle time (not average), because the timeout must always be based on the worst case scenario. Example: average cycle time 4 seconds / worst cycle 6 seconds → timeout 18-30 seconds.
### 4.3 Spatial reasoning findings

- Introduction of additional spatial structure and pixel/region-level prompts resulted in almost doubling of accuracy compared to unstructured prompts.
- Prompting alone failed to be enough in cases of complex images – hit a plateau before reaching full reliability.
- The most efficient method was the **hybrid use of OpenCV together with the LLM**, allowing OpenCV to deal with geometric  directly, which is more efficient than letting VLM infer pixel coordinates itself.
- "Anchor point" prompting (providing the model with a certain reference frame) helped to stabilize results across cycles.

---

## 5. Open Questions / Next Steps 

Determine when/how to bring back Isaac Sim for simulation-based evaluation after establishing stability of the hardware loop.
- Work on optimizing REACQUIRE/scan recovery functionality for robustness.
- Literature review: see the Related work section of [`paper/paper.md`](../../../paper/paper.md#2-related-work).
- Richer reasoning enhancements suggested as future work instead of replacing VLA: memory of previous sightings/perspectives, failure history-informed searching strategy, adaptable scan pattern.
