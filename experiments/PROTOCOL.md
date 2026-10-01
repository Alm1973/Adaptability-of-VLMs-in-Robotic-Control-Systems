# Protocol: controlled-rig experiment, 1 September 2026

This is the pre-registration for run `2026-09-01_233533`, written out in one place. Everything in the first four sections was fixed in `experiment_config.json` and in the header and scoring code of `final_experiment.py` before the first trial. Both files are saved with the run in `results/2026-09-01_final/`. Changes made after the run are listed separately at the end.

## 1. Hypotheses

- **H1 (primary).** The VLM's contribution is confined to refusing false presence: rejecting a substituted or same-class object standing where the target was. It adds nothing measurable to holding belief through ordinary occlusion.
- **H0 (null).** Removing the VLM changes neither false-belief frames nor during-disruption accuracy on any disruption type.
- **H2 (secondary).** Occlusion recovery comes from the belief state machine's decay timer, not the VLM, so occlusion failures depend on how long the disruption lasts relative to the timer, not on what the occluder is.

These came from the August pilot, where the full system and the no-VLM version disagreed on 0 of 515 frames on the environment and unexpected-object episodes.

## 2. Design

**Live factors (staged by the operator):**

- Scenario (8): control, occlude_hand, occlude_object, removal, substitution, identity_swap, lighting, camera_pose.
- Clutter (2): *none* is the bare rig with only the red cup on its marked position. *High* is six to eight other objects in frame, including at least two other cups or mugs of different colors, placed around the cup without covering it. The same objects were used all session.

**Offline factor (applied to the recorded frames):** the ablation. Every frame is replayed through the full system (`D_full`) and through the same system with the VLM removed (`E_no_vlm`), and through `D_full` with the decay timer set to 30 and 60 frames. Replaying the same frames removes staging variation from the comparison.

**Size and order:** 3 repetitions of every scenario at each clutter level, 48 trials. Order is counterbalanced by block with seed 20260901.

**Held constant:** camera settings, camera pose (arm homed to 90°/90° before and after every trial), lighting, target position, occluder set, phase durations, and the target string "red cup".

## 3. Procedure and ground truth

Each trial has three 6-second phases: baseline, disrupt, recover. The script announces each phase and its ground truth (cup present or absent), waits for the operator to confirm the scene is set, then records. Ground truth comes from the script's instruction at capture time and is never assigned by reviewing footage.

Frames within 1 s of a phase boundary are marked `edge`. They go through the pipeline so the belief stays continuous, but they are not scored. A trial the operator marks as badly staged is invalid: it is logged, not silently dropped, and restaged.

## 4. Measures and pass rule

| Measure | Definition |
|---|---|
| False-belief frames (primary) | Scored frames where the system says the cup is present and it is absent |
| During-disruption accuracy | Share of scored frames during the disruption where the belief matched the truth |
| Lost while present | Scored frames where the cup was present but the system did not believe it |
| Spurious reacquisition | The trial ends CONFIRMED although the cup is gone |

False belief is never averaged into accuracy. In the pilot, removing the VLM raised accuracy on one identity episode (0.615 to 0.923) while false belief rose from 5 to 23 frames, so ranking by accuracy alone would have picked the more dangerous pipeline.

| Verdict | Rule |
|---|---|
| PASS | accuracy ≥ 0.70 and zero false-belief frames |
| PARTIAL | accuracy ≥ 0.40 and zero false-belief frames |
| FAIL | anything else, or any false-belief frame at all |

**Planned analysis:** compare `D_full` with `E_no_vlm` on the replay, report the decay trade-off, and report per-scenario results as descriptive only (n = 3 per cell).

## 5. What changed after the run

1. **Accuracy window.** The pre-registered definition is accuracy "while the disruption is active". The live scorer in the script that ran also counted the recover phase, which is easy and inflated the live numbers. The replay scorer uses the disrupt phase only, and all reported accuracy and verdicts use the replay. The live verdicts (25 pass, 13 partial, 10 fail) are kept for reference. Under the pre-registered definition the 13 partials become failures. `avi/final_experiment.py` now uses the disrupt phase only.
2. **Restaged trials.** Trials 3 (occlude_hand) and 6 (substitution) were restaged at the operator's request before scoring. Both restagings are in `run_log.txt` and the discarded attempts were not scored.
3. **Statistics added after the run:** Wilson 95% intervals on detector false-positive rates, Fisher's exact test for bare vs. clutter, and a sign test on the trials where the two conditions differed. The sign test's one-sided p is reported as the planned direction (the VLM reduces false belief), and the two-sided p is reported too.

## 6. Proposed extension (not run)

About 24 more trials covering only the scenarios where the cup ends up gone (removal, substitution, identity_swap; 8 repetitions each), logging the detector's box on every frame. The replay would also test a stricter detection threshold, periodic re-checks while CONFIRMED, and a longer hold in OCCLUDED than after signs of a swap. Any extension will be written here, dated, before it runs, and reported separately from the 1 September run.
