# AVI Robot Project — Complete Report (Aug 4–8, 2026)

> Source: Google Doc · (private Google Drive link removed)
> Created 2026-08-08 · Last modified 2026-08-08 · Imported 2026-09-28 (verbatim)
> Duplicate: the shared doc "Current progress " (private Google Drive link removed) last edited 2026-09-28) has the same content with trivial wording differences — not imported separately.
> Referenced figures (not in Drive text): report_latency_evolution.png, report_accuracy_evolution.png, live_experiment_chart.png. — likely on the MSI or Mac.

**Period:** August 4 – August 8, 2026 | **Hardware:** MSI laptop (RTX 3060 Laptop 6 GB VRAM, driver 610.62), Arduino Uno on COM5 (base + tilt servos), Logitech C270 camera, controlled from MacBook via Tailscale/SSH | **Models:** qwen2.5vl:3b (primary), moondream (rejected), via llama-server on GPU

Figures: report_latency_evolution.png, report_accuracy_evolution.png, live_experiment_chart.png.

| | Start (Aug 6 baseline) | End (Aug 8) |
| :- | :- | :- |
| **Detection latency / frame** | 65.9 s (per-object VLM + degenerate bug) | **8 ms** (OpenCV continuous path) — ~8,200× |
| **VLM calls / frame** | 3–5 | **0.04–0.19** |
| **Hallucinations (held-out, 40 probes)** | 6 (bicycle ×3, banana ×2) | **0** |
| **Held-out accuracy** | 4/8 (0.50) | **0.771** (majority-of-3) |
| **'@@@@' corruption reaching output** | frequent, unexplained | **never** (restart ladder + rotation) |
| **Capabilities** | fixed red-cup tracker | open-vocabulary: findobject, findsearch (servo hunt + follow), livetrack, typo correction |

| Date | What happened |
| :- | :- |
| Aug 4 (sessions 1–8) | First '@@@@' degenerate GPU output. Session 6: full forensic (driver/VRAM/TDR exonerated; per-request streaky, upstream llama.cpp #14663 match). Session 7: built llm_backend.py managed GPU llama-server; found switch to the silent CPU fallback trap. |
| Aug 6 (session 9) | Write up |
| Aug 6–7 (night) | Teleop (WASD + MJPEG stream), passwordless SSH, runhome command, Windows Task Scheduler cycles at 2:10 AM + every 4 h. |
| Aug 7 (all day/night) | The '@@@@' root-cause is from of controlled experiments downloads + mitigation. 22× speedup shipped. |
| Aug 8 (early) | Improvement loop: open listing (0 hallucinations) |
| Aug 8 (overnight, autonomous) | corrected the encode-budget assumption (corruption is bimodal, locks at call 3–4). |
| Aug 8 (day) | New architecture: VLM-configures-then-verifies + OpenCV continuous. Live servo experiment. findsearch final: search - find - verify - servo-follow loop. Red-cup tracking bugs found and fixed. Typo correction added. |

## 1. GPU Vision Model Issue
The biggest problem during development was that the vision model would randomly stop working on the GPU. Instead of giving a normal answer, it would return meaningless characters which arent usable at all. When this happened the system had to switch to the CPU, making responses much slower.

After extensive testing, almost every possible cause was ruled out, including:
- Different sampling settings
- Context sizes
- CUDA options
- Batch sizes
- VRAM limits
- Different model builds

The only thing that had an impact on this issue was the way the image was processed by the GPU. Big images or newly encoded images would cause the GPU vision pipeline to fail at some point.

Previous "successful" tests for stability of the system proved to be misleading as well, since all the images used were previously tested. The model actually relied on image embeddings from cache, rather than processing a brand new one.

### Final conclusion
However, there has been no total solution so far.

Rather than attempt to tackle the problem of the bug itself, the software now checks for corrupted responses, restarts the vision backend when necessary, and resorts to the use of the CPU when necessary.

# Avoiding Misleading Test Results
- A number of experiments initially seemed promising, but they ended up being misleading.
- For instance:
- Using the same image for tests masked the problem with GPU since the model worked with the cached images.
- Evaluation of the performance after recovery system exaggerated the problems.
- The improvements of prompts worked only for the test images they were trained on and not the others.
- In light of this, several guidelines have been set for all further tests:
- Always test the model using the new images.
- Do not measure performance using the recovery system, measure the performance of the model.
- Evaluate any changes using separate validation data.

# Reducing Hallucinations
Initially, questions posed by [the system] would be like:
"Is there a red cup?"
Which inadvertently led to the model generating objects that were not even there.

The way around this was to ask first:
"What objects do you see?"
After which the response is compared to the desired object requested by the user.
This helped in reducing false positives, as well as improving accuracy.
[It] also repeated the running of the vision model several times and selected the answer that appeared most often.

# New Detection Pipeline
In lieu of the vision model analyzing all the frames, [the system] delegates the task.

The vision model:
- Examines the scene one time.
- Discovers the appearance of the object.
- Configures the tracking parameters.

OpenCV:
- Traces the object in real time on a continuous basis.
- Requests another examination from the vision model only if there is a problem with the tracking or there is some anomaly.

This significantly lowers the number of vision model requests and speeds up the tracking process.

# Real-World Problems Found
**Color recognition** — The vision system sometimes referred to red objects as blue. [It] now always relies on the user-defined color name when a color is provided.

**Object mislocation** — At times the vision system pointed to the wrong place in the picture. [It] now validates the correct colors and recalibrates if necessary.

### Red color detection
Because red wraps around the HSV color space, a simple color range caused many false detections. [It] now measures red correctly using two HSV ranges.

### Tracking jumps
The tracker occasionally switched between similar-looking objects. [It] now remembers the previous object position and only changes targets after verification.

### Bad detector settings
If the automatically generated tracking settings found nothing, the robot previously became stuck. Now it automatically reconfigures itself and tries again.

# Final Search and Tracking System
The implemented search system will:
- Sense the environment automatically.
- Find the desired object.
- Configure tracking parameters.
- Track the object effortlessly by using OpenCV.
- Recenter the object by using the robotic arm.
- Continue the search automatically if the object gets lost.

# 7. Other Improvements
Other additional features implemented during development include:
- Remote control of the robot through the network.
- Scheduling overnight research experiments for automatic testing of hypotheses.
- Protections against accidental movement of the robotic arm without a person present.
- Control commands for search, tracking, experimentation, and teleoperation.
