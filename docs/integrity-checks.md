# Integrity checks — run `2026-09-01_233533`

> **Corrections (added 2026-09-28, after re-checking the raw files).** Two items below are wrong. Use these instead:
>
> 1. **Item 4 (0.768 vs 0.640).** This is not a disagreement between scorers. The live scorer counted the disrupt and recover phases; the replay counted the disrupt phase only. Recomputing both windows from `results.json` reproduces both numbers exactly. The paper reports the disrupt-only value (0.640) and explains the difference in Section 4.5.
> 2. **Item 6 (VLM runtime).** The 1 September run did not use llama-server. `final_experiment.py.snapshot` loads the VLM through `run_study.make_fast_verifier`: Qwen/Qwen2.5-VL-3B-Instruct in Hugging Face transformers with 4-bit NF4 weights, images capped at 100,352 pixels, answered from the first-token yes/no scores. llama-server and Ollama were the August stack.
>
> Paths in this file refer to the laptop's folder layout. In this repository the run is in `results/2026-09-01_final/`.


All numbers below were computed read-only from the raw files copied into
`avi/runs/2026-09-01_233533/` (`results.csv`, `results.json`,
`ablation_scores.json`, `run_log.txt`, `experiment_config.json`,
`final_experiment.py.snapshot`) plus the copied source modules
(`arm.py`, `config.py`, `yolo_tracker.py`, `llm_backend.py`,
`recovery_pipeline.py`). No script that touches the camera, arm, or serial
port was executed — only read, parsed, and grepped as text/JSON/CSV. The
analysis script used is reproduced inline per check so the arithmetic is
auditable; it is not committed separately.

---

## 1. Real camera frame rate: ~28 fps (reports) vs ~10 fps (actual)

**~10 fps is the real capture rate. ~28 fps is not a frame rate at all —
it's `1000 / mean(step_ms)`, the reciprocal of average per-step processing
time, mislabeled as a capture rate.**

Computed from `results.json`, which stores an elapsed-time field `t` (seconds
since trial start) on every one of the 8,428 logged frames across 48 trials:

- Consecutive-frame intervals within each trial (8,284 gaps):
  **mean Δt = 0.1035 s → 9.66 fps, median Δt = 0.1000 s → 10.00 fps.**
  The median is exact: frames were captured on a fixed 10 Hz tick.
- `step_ms` (per-frame pipeline processing time, also logged per frame):
  mean = 34.6 ms → 1000/34.6 = **28.90 fps**; median = 29.0 ms → 34.5 fps.
  The mean figure (28.9) is what the reports rounded to "~28 fps" — but
  `step_ms` measures how long one pipeline step took to *compute*, not the
  interval between frames being *captured*. It is pulled down toward ~29 ms
  by the large share of cheap non-VLM steps and is a different quantity from
  the fixed 10 Hz capture cadence entirely.
- `total_frames / wall_clock_s` per trial (`results.csv`) gives 3.6–3.8 fps,
  even lower — because `wall_clock_s` includes settle time and phase
  transitions, not just active capture, so this ratio isn't the right
  denominator either. It's included here only to show it does NOT support
  the ~28 fps figure.

**Conclusion: report ~10 fps as the frame rate; the ~28 fps figure should be
relabeled or dropped — it describes processing speed, not capture rate.**

---

## 2. D_full false-belief frame sum: 196 (reports) vs 193 (per-trial table)

**Both numbers are real, they measure different things: 193 is the LIVE run's
own scoring, 196 is the OFFLINE REPLAY (ablation) scoring of the same
captured frames.**

```
results.csv sum(false_belief_frames), all 48 trials (LIVE run):        193
ablation_scores.json per_episode["D_full"] sum("false"), 48 eps
  (OFFLINE REPLAY via recovery_pipeline.py, decay_frames=12):          196
ablation_scores.json per_episode["E_no_vlm"] sum("false") (replay):    297
```

`final_experiment.py.snapshot`'s own docstring explains the design: "Capturing
once and scoring every ablation offline is deliberate... it removes the
operator's staging variance from the D-vs-E comparison entirely." The live
run's real-time state machine and the offline `recovery_pipeline.py` replay
are two independently-executed scorers over related but not identical frame
streams/state-machine parameters, so small per-trial disagreements
(confirmed below in check 4) accumulate to a 3-frame difference in the sum.
**The reports' 196 is the replay/ablation number; the per-trial CSV table's
193 is the live number.** Neither is wrong — they should be labeled by
source, not merged.

---

## 3. D_full vs E_no_vlm on false belief: 6 trials differ (not 7); trial 36 is NOT one of them

```
009_identity_swap_rep1:   D_full=0  E_no_vlm=26  (Δ=+26)
012_substitution_rep1:    D_full=6  E_no_vlm=9   (Δ=+3)
014_removal_rep1:         D_full=0  E_no_vlm=37  (Δ=+37)
029_removal_rep2:         D_full=15 E_no_vlm=36  (Δ=+21)
030_identity_swap_rep2:   D_full=19 E_no_vlm=22  (Δ=+3)
045_identity_swap_rep3:   D_full=17 E_no_vlm=28  (Δ=+11)
```

That's **6 episodes**, all with E_no_vlm strictly worse (more false belief) —
never the reverse, never tied-then-different. `036_removal_*` is not in the
list (D_full and E_no_vlm agree on episode 36).

**Sign test** (H0: VLM removal is equally likely to help or hurt
false-belief count; two-tailed exact binomial on the 6 non-tied episodes,
p=0.5 each):

- 6 of 6 differing episodes favor D_full (VLM present) having fewer/equal
  false-belief frames; 0 favor E_no_vlm.
- One-sided exact p = P(X ≤ 0 | n=6, p=0.5) = (6 choose 0)/2⁶ = 1/64 =
  **0.0156**
- Two-sided exact p = min(1, 2 × 0.0156) = **0.0312**

---

## 4. D_full "mean acc" 0.640 vs per-trial average 0.768

**0.768 is the simple mean of `during_disruption_acc` from the LIVE run
(`results.csv`, 48 trials). 0.640 is the simple mean of `durAcc` from the
OFFLINE REPLAY (`ablation_scores.json`, `per_episode["D_full"]`, 48
episodes).** Same live-vs-replay split as check 2, not a weighting artifact:

```
results.csv simple per-trial mean of during_disruption_acc (LIVE):     0.7682
ablation_scores.json D_full simple per-episode mean of durAcc (REPLAY):0.6402
D_full durAcc weighted by live scored_frames/trial (sanity check):     0.6421
corpus['scored']/corpus['frames'] (all-frames pooled ratio, distinct
  quantity — fraction of captured frames that were scored at all):     0.6762
E_no_vlm simple per-episode mean durAcc (REPLAY), for reference:       0.6526
```

Re-weighting the replay mean by the live run's per-trial scored-frame counts
(0.6421) barely moves it from the flat mean (0.6402) — so the 0.768-vs-0.640
gap is not a frame-weighting effect. It is the same live-scorer-vs-replay-
scorer disagreement documented in check 2: the two scoring passes disagree
per-trial (e.g. trial 2: live acc 0.532 vs replay durAcc 0.051 — a
~10x difference for that single episode), and those disagreements don't
cancel out over 48 trials.

**Report 0.768 as the live run's headline accuracy and 0.640 as the offline
D_full-condition (decay=12) replay accuracy; do not average or substitute
one for the other.**

---

## 5. Trial 21: acc 1.0, but 34 "lost" frames — `lost_while_present` is counted over the WHOLE trial, not just the disruption window

`results.csv` row 21: `during_disruption_acc=1.0`, `lost_while_present=34`,
`scored_frames=114` (of `total_frames=170`, `edge_frames=56`).

Breaking `results.json` trial 21's 170 per-frame records down by phase and
by `edge`/`status`:

```
frames per phase:            baseline=51, disrupt=61, recover=58
scored (non-edge) per phase: baseline=34, disrupt=40, recover=40   (sums to 114 ✓)
"lost" (present, non-edge, status != CONFIRMED) per phase:
    baseline: 34   disrupt: 0   recover: 0
```

All 34 lost-while-present frames are scored frames from the **baseline
phase** (mostly `status="AMBIGUOUS"`, a few `MISSING`) — before the
disruption for trial 21 (`occlude_object`, rep 2) even starts. Zero lost
frames occur in `disrupt` or `recover`.

`during_disruption_acc` is scored only over the `disrupt`+`recover` phases,
so it correctly shows 1.0 (perfect belief during the actual disruption).
`lost_while_present` is a whole-trial (all scored phases, including
baseline) counter, so it correctly shows 34. **They're not inconsistent —
they're windowed differently by design**, and trial 21 happens to be a case
where all the loss occurred in a window that the accuracy metric excludes.
This is worth flagging in the paper as a metric-definition footnote, since a
reader skimming the table would reasonably expect "1.0 acc" and "34 lost
frames" to be talking about the same window.

---

## 6. Configuration, from `final_experiment.py.snapshot` and the modules it imports

`final_experiment.py.snapshot` itself defines only trial orchestration
constants (`HOME=(90,90)`, `PHASE_SECONDS=6.0`, `EDGE_SECONDS=1.0`,
`PASS_ACC=0.70`, `PARTIAL_ACC=0.40`); it imports the arm, detector, and VLM
backend from separate modules (`from yolo_tracker import YoloTracker` at
line 789), so the answers below come from those, all copied alongside it
under `avi/`:

- **Arm DOF / servos** (`arm.py`): 2 DOF — a `base` (pan) servo and a `tilt`
  servo, each clamped to 0–180°, both homed to 90/90. Communicates over
  `serial.Serial('COM5', 115200, ...)` sending `"X:<Δbase> Y:<Δtilt>\n"`
  relative-move commands. No gripper/other axis is present in this class.
- **VLM serving** (`llm_backend.py`): `llama-server.exe` (the llama.cpp
  server binary bundled with the local Ollama install), launched as a
  subprocess with `-ngl 99` (full GPU offload), `-c 2048`, flash-attn off,
  serving an OpenAI-compatible `/v1/chat/completions` endpoint on
  `127.0.0.1:18080` (`LLAMA_SERVER_PORT` in `config.py`). Not "plain Ollama
  serve" — it launches llama-server directly against an Ollama-managed model
  blob, with a CPU-mode `ollama chat` (Python `ollama` package) fallback
  path (`_ollama_cpu_chat`) for GPU-degenerate recovery.
- **Model file + quantization**: the blob loaded is Ollama's on-disk blob
  for tag `qwen2.5vl:3b`. `ollama show qwen2.5vl:3b` (read-only query,
  run today) reports: architecture `qwen25vl`, 3.8B parameters,
  **quantization Q4_K_M**, context length 128000, vision capability, 3.2 GB
  on disk. `OLLAMA_FALLBACK_MODEL = "qwen2.5vl:3b"` in `llm_backend.py`
  matches.
- **YOLO-World weights** (`yolo_tracker.py`): `DEFAULT_MODEL =
  "yolov8s-worldv2.pt"` (also present at the top of the laptop's home
  directory, ~25 MB — not copied into the repo per the model-weights
  exclusion).
- **Detector confidence threshold** (`yolo_tracker.py`): `MIN_CONFIDENCE =
  0.25` (active threshold while not locked onto a target); `HOLD_CONFIDENCE
  = 0.15` (lower threshold used once a target is already locked, to resist
  momentary confidence dips without re-triggering acquisition logic).
- **DECAY_FRAMES**: `recovery_pipeline.py` (the module `score_run.py` /
  the ablation scorer actually uses) defines `DECAY_FRAMES = 12` — this is
  the D_full default; `ablation_scores.json`'s extra conditions
  `D_full(decay=30)` and `D_full(decay=60)` are explicit sweeps off that
  default. (Note: `find_live.py`, a different tool for the search/live-track
  utilities — not the Sept 1 experiment — defines its own unrelated
  `DECAY_FRAMES = 8`; don't conflate the two.)
- **Camera resolution** (`config.py`): `FRAME_WIDTH = 1280`,
  `FRAME_HEIGHT = 720` (`CAMERA_INDEX = 1`).

---

## 7. Software versions (read-only queries, run 2026-09-27 — see caveat)

```
Python:            3.13.13   (.venv-yolo/Scripts/python.exe --version)
torch:              2.13.0+cu126   (CUDA available: True, CUDA 12.6)
ultralytics:        8.4.116
opencv-python (cv2): 5.0.0
ollama:              0.34.0   (ollama --version)
GPU / driver:        NVIDIA GeForce RTX 3060 Laptop GPU, driver 610.62
                     (nvidia-smi --query-gpu=name,driver_version --format=csv)
```

**Caveat:** none of `run_log.txt`, `experiment_config.json`, or
`final_experiment.py.snapshot` record software versions at run time, so the
above are the *current* environment on this laptop (queried 2026-09-27,
~26 days after the run), not a frozen snapshot from Sept 1. If exact
run-time versions matter for the paper's reproducibility section, this is a
gap worth closing going forward (e.g. have the harness dump `pip freeze`
and `ollama --version` into each run directory) — noted as a suggestion, not
retroactively fixable from read-only inspection.
