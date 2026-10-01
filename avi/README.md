# avi: the final system

This folder holds the code that ran the 1 September 2026 experiment. Run scripts from inside this folder (`cd avi`), because the modules import each other by name.

## Modules

| File | What it does |
|---|---|
| `recovery_pipeline.py` | The belief state machine (CONFIRMED, OCCLUDED, DISPLACED, MISSING, AMBIGUOUS), the OpenCV change checks that run on lost frames, and the rule for when to ask the VLM |
| `yolo_tracker.py` | YOLO-World wrapper (`yolov8s-worldv2.pt`), prompted with the target name |
| `run_study.py` | Loads the VLM. `make_fast_verifier()` runs Qwen/Qwen2.5-VL-3B-Instruct in 4-bit NF4 and answers yes/no from the first-token scores. Also scores episodes offline |
| `disruption_bench.py` | Episode helpers shared by the study and replay scripts |
| `final_experiment.py` | The staged experiment: 8 scenarios × 3 reps × 2 clutter levels, operator prompts for each phase, frame capture, live scoring |
| `score_run.py` | Offline replay: rebuilds episodes from `results.json` and scores them under each ablation condition. Writes `ablation_scores.json` |
| `test_final_experiment.py` | Self-checks for trial ordering, counterbalancing and scoring |
| `preflight.py` | Pre-run checks: code imports, trial order balance, free GPU memory and disk, COM port, camera frame rate, detector sees the target, one pipeline step |
| `make_figures.py` | Builds the contact sheets and videos in `media/` from a run folder |
| `arm.py`, `home.py`, `teleop.py` | Serial control of the arm, re-centering, and a small web page for manual control |
| `camera.py`, `config.py` | Camera capture (1280×720) and shared settings |

## Which `final_experiment.py` ran

`final_experiment.py` in this folder includes one change made after the run: during-disruption accuracy is computed over the disrupt phase only. The script that produced the 1 September results used the disrupt and recover phases together. It is saved, with its original comments, as `results/2026-09-01_final/final_experiment.py.snapshot`. The paper explains the change in Section 4.5.

## Before you run anything on hardware

- Opening the serial port resets the Arduino, and the firmware then drives both servos to center. Keep hands and cables clear.
- The port name is hard-coded as `COM5` in `arm.py`, `home.py` and `preflight.py`. Change it for your machine.
- The run needs about 3.8 GB of free GPU memory. `preflight.py` fails below 4,000 MiB, which usually means another model is still loaded.
- Captured frames go into `avi/runs/<timestamp>/`. That folder is git-ignored, because a full run is about 3 GB.
