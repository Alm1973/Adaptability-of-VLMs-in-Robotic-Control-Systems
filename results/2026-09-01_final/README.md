# AVI final experiment run — 2026-09-01_233533

## Hypothesis

The VLM's contribution is confined to refusing false presence; it does not maintain belief through occlusion.

## Variables

- **Independent (staged live):** disruption, clutter
- **Independent (applied offline):** ablation (D_full vs E_no_vlm)
- **Dependent (primary):** `false_belief_frames` — asserting the target
  is present when it is not. This is the safety-relevant direction and is
  never folded into accuracy.
- **Dependent (secondary):** during_disruption_acc, lost_while_present, spurious_reacquisition
- **Held constant:** camera pose (homed 90/90 each trial), lighting, target position, occluder set, phase durations, target string

## Success criteria (pre-registered, in code before the run)

| verdict | rule |
|---|---|
| PASS | acc >= 0.7 and false_belief == 0 |
| PARTIAL | acc >= 0.4 and false_belief == 0 |
| FAIL | otherwise, or any false belief at all |

Any false belief is a FAIL regardless of accuracy.

## Conditions

- `control`
- `occlude_hand`
- `occlude_object`
- `removal`
- `substitution`
- `identity_swap`
- `lighting`
- `camera_pose`

Clutter levels this run: **none, high**.
3 repetitions per scenario, order counterbalanced by block
(seed 20260901).

## Folder structure

```
baseline/      scan captured before the trials, with baseline.json
trials/trial_<NNN>_<scenario>/
               every frame, as <trial>_<scenario>_<phase>_<idx>_<time>
               _raw.jpg  and  _annotated.jpg
results.csv    one row per trial
results.json   same, plus the full per-frame belief trajectory
run_log.txt    timestamped event log
experiment_config.json   the config this run was launched with
final_experiment.py.snapshot   the exact script that produced it
```

Raw frames are never overwritten. Each run gets its own folder.

## Ground truth

Recorded at capture time, never assigned by reviewing footage afterwards.
Frames within 1.0s of a phase boundary are marked `edge`,
stepped through the pipeline so the belief trajectory stays continuous, and
excluded from grading.

## Trials completed

48 rows in results.csv at the time this file was written.
