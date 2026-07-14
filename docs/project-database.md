# AVI Project Database

Extracted and organized from ChatGPT conversation export ("Hybrid AI Robot Setup"). This document consolidates architecture decisions, calibration data, and system design discussion for the Adaptability of VLMs in Robotic Control Systems (AVI) project.

---

## 1. Research Question & Framing

**Working research question:**
> Can a locally run VLM autonomously recover from task-disrupting situations — occlusion, environmental change, and abnormal/anomalous targets — through dynamic reasoning and adaptive camera repositioning, during an object verification task?

Earlier framing iterations considered before settling on the above:
- *"How effectively can an LLM-guided robotic system recover from task disruptions that were not present in its training data?"*
- A three-part failure taxonomy was defined early and kept throughout: **occlusion**, **environmental change**, **anomaly** (unexpected/abnormal target objects).

**Metrics identified for evaluating success:**
- Recovery time (seconds to adapt)
- Task success rate (%)
- Number of retries
- Error type classification (perception, planning, actuation)

**Key methodological distinction drawn (important for the paper):** the project is explicitly studying a *reasoning loop* (see → reason → plan → act → observe → reason again), not an end-to-end action-prediction policy (VLA-style). This was used to justify **not** switching to a Vision-Language-Action model like π0 or OpenVLA — doing so would change the research question from "can a model reason its way through disruption" to "can an end-to-end policy control a robot," which is a different scientific question.

---

## 2. System Architecture

### 2.1 Architecture decision (Architecture A vs. B)

Two candidate architectures were formally evaluated:

**Architecture A — "Pixy2 sees, LLM thinks"**
`Pixy2 (onboard color-signature detection) → Arduino → Raspberry Pi → small text LLM (Qwen 0.5B–3B, Ollama)`
- LLM never sees an image — only structured text like `Object detected: signature=2 (red), x=152, y=88...`
- Pros: uses existing hardware, lower compute, novel non-visual spatial-reasoning angle.
- Cons: Pixy2 limited to color blobs — can't represent shape/scene detail, weakens the "anomaly" failure category.

**Architecture B — "VLM sees and thinks"**
`Camera (webcam/Pi Camera) → VLM (image input) → reasoning + action, on laptop or cloud API`
- Pros: rich perception, natural fit for anomaly detection, no Pi needed.
- Cons: doesn't run at usable speed on a Raspberry Pi 5 — needs laptop or cloud compute.

**Decision:** The project moved forward with a **VLM-based approach** (Architecture B direction), ultimately running **Qwen2.5-VL:3B via Ollama** locally rather than a cloud API, with a USB webcam (Logitech C270, 720p) and Arduino for actuation. (Note: later project direction per current memory has since moved this reasoning model onto a Raspberry Pi 5, consistent with keeping compute local/on-device.)

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

**Rationale (explicitly discussed and defended):** OpenCV vs. adaptability is a false tradeoff. OpenCV is a deterministic perception front-end; adaptability comes from the decision/policy layer above it, not from how pixels are turned into coordinates. The reasoning stack is layered as:

- **Layer 1 — Perception:** OpenCV baseline (HSV masking + centroid), swappable later for YOLO/SAM/VLM embeddings.
- **Layer 2 — State tracker:** object position relative to center, confidence, occlusion level, motion history/smoothing.
- **Layer 3 — Policy (the actual research contribution):** rule-based/heuristic → later RL or VLA-style.
- **Layer 4 — Actuation:** 4-DOF servo arm control via Arduino.

Planned phased rollout:
- **Phase 1:** OpenCV → object centroid → move arm to center (baseline, deterministic).
- **Phase 2:** Replace/augment OpenCV with YOLO or VLM perception.
- **Phase 3:** Replace rule-based policy with RL or VLA.

### 2.3 VLA / π0 evaluation (rejected for now)

π0 (Physical Intelligence) and OpenVLA were investigated as possible replacements for the whole pipeline. Conclusion: **not adopted**, because:
- π0-class models target 6–7 DOF arms with force feedback, joint encoders, and grippers — AVI has 4 servos, no encoders, no force sensing, and a pure visual-search task (no manipulation).
- Replacing the reasoning loop with an end-to-end VLA would eliminate the exact thing being studied (explicit multi-step reasoning/recovery), collapsing it into "see → act."
- Recommended alternative direction: make AVI's *reasoning* richer instead (memory of past viewpoints, hypothesizing where a lost object might be, scene-conditioned search-strategy selection) — this stays aligned with the current research question rather than pivoting to a different one.

### 2.4 Simulation (Isaac Sim) — parked

Isaac Sim / URDF import work was attempted for a Panda-arm simulation environment but hit substantial tooling friction (version mismatches, missing UI tools, USD joint-path discovery). Decision: **paused** in favor of the lightweight real-hardware OpenCV + VLM loop, which gives faster iteration. Isaac Sim work may be revisited later once the core perception → reasoning → action loop is stable on real hardware.

---

## 3. Hardware

- Raspberry Pi 5 (evaluated in Architecture A; current direction favors Qwen running on-device per most recent project state)
- Arduino (Mega, later using a PCA9685 16-channel PWM driver for servo control)
- 4× DS3218 20kg·cm digital metal-gear servos (270°, waterproof), used for base / shoulder / elbow / wrist (or camera pan-tilt)
- UBEC for servo power (must be rated ≥10A, ideally 15A, given up to ~9A combined spikes from 3× 20kg servos under load)
- Logitech C270 webcam, 720p, USB — chosen for lightweight mounting and adequate resolution at a 25–35cm working height over a 150cm×150cm workspace
- Common ground between UBEC and Arduino is a critical wiring requirement (identified as the most common failure point)

### 3.1 Servo calibration (final values)

Per-servo calibration replaced an earlier single global PWM mapping (`map(angle, 0, 180, 75, 580)` — kept only as the original/default assumption).

| Servo | Role | MIN (PWM) | CENTER (PWM) | MAX (PWM) |
|---|---|---|---|---|
| Servo 0 | Base rotation | 75 | 250 | 460 |
| Servo 1 | Shoulder / tilt | 75 (front down) | 370 (up) | 580 (back down) |
| Servo 3 | Wrist | 80 | 430 | 570 |
| Servo 4 | (4th joint) | 140 | 380 | 580 |

Home position (early setup): `base=55, shoulder=115, elbow=125, wrist=110`.

---

## 4. Software / Codebase

The working system was refactored into a modular Python package (`avi/`) plus Arduino firmware:

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

**State machine behavior:** `SEARCHING → TRACKING → REACQUIRE`, with a scan strategy while searching/reacquiring and color-tracking while a target is locked. Debugged issue: system could get permanently stuck in `REACQUIRE` if there was no guaranteed exit condition back to `TRACKING` on a fresh detection, and if the "scan" motion command wasn't wired to real Arduino movement — both were identified and patched (raise the missed-frame threshold before entering REACQUIRE, add an immediate `REACQUIRE → TRACKING` transition on detection, and replace the no-op scan with actual small alternating base movements).

### 4.1 Object tracking (red-cup baseline)

Baseline tracking loop (`vision.py`-equivalent): HSV dual-range red mask (`0-10` and `170-180` hue bands) → largest contour above an area threshold → bounding-box centroid → error from frame center → proportional base/wrist servo nudge (deadzone ~40px, send interval ~80ms) → serial write of `BASE:<angle>` / `WRIST:<angle>` commands to Arduino, which parses and drives the PCA9685.

### 4.2 Latency benchmarking results

From assignment-style benchmarking work on the reasoning stage:

| Approach | Latency | Note |
|---|---|---|
| CPU-based LLaVA (baseline) | ~67.8s | Too slow for real-time robotics |
| GPU-accelerated via MLX | ~12.0s (later reported ~6s) | Major improvement, still not fully real-time |
| Moondream (lightweight local model) | ~4.8s | Fastest local option tested |
| Llama.cpp runtime | faster than Ollama | Lower-level runtime reduces overhead |
| Cloud API | fastest overall | Adds network dependency + minor per-call cost |

Only lightweight local models and cloud APIs were found to meet real-time-ish constraints; general local model inference was the dominant bottleneck, not the vision or actuation stages.

**Timeout calibration methodology:** full loop cycle = observe → decide → move → re-check. Recommended timeout = 3–5× the *worst observed* cycle time (not average), since timeouts must be based on worst-case behavior. Example: avg cycle 4s / worst cycle 6s → timeout range 18–30s.

### 4.3 Spatial reasoning findings

- Adding explicit spatial structure/pixel-region rules to prompts roughly doubled accuracy over an unstructured prompt.
- Prompting alone was insufficient in complex scenes — plateaued below full reliability.
- The **hybrid OpenCV + LLM approach performed best**: letting OpenCV handle geometry directly and removing spatial guesswork from the model was more accurate than asking the VLM to infer pixel coordinates itself.
- "Anchor point" prompting (giving the model a fixed reference frame) improved cross-cycle stability.

This directly supports the current architecture's grid-classification / HSV-centroid-to-text-label design: perception stays deterministic (OpenCV), and the VLM only reasons over compact structured text rather than raw pixel coordinates.

---

## 5. Open Questions / Next Steps (as of last captured conversation)

- Decide whether/when to reintroduce Isaac Sim for simulation-based testing once the real-hardware loop is stable.
- Continue tuning REACQUIRE/scan recovery logic for reliability.
- Formal literature review (VLM failure recovery, edge LLM deployment, active perception, camera repositioning) exists in prior sessions but was not present in this export — see main memory summary for the ~27-paper ranked list; needs to be pulled from ChatGPT separately if a standalone `docs/literature-review.md` is wanted.
- Possible richer-reasoning upgrades identified as future work rather than a VLA swap: memory of past detections/viewpoints, failure-history-conditioned search strategy, adaptive scan patterns.

---

*Source: single ChatGPT conversation export ("Hybrid AI Robot Setup"), covering sessions roughly April 9 – July 2, 2026. This is not exhaustive of all AVI-related ChatGPT conversations — only what was exported and uploaded.*
