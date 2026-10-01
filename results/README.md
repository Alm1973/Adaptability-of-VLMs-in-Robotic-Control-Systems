# Results

| Path | What it is |
|---|---|
| `2026-09-01_final/` | Raw output of the pre-registered run (48 trials, 8,428 frames), exactly as the lab laptop wrote it, plus a sample of annotated frames |
| `analysis/analyze.py` | Recomputes every number, table and figure in the paper from the raw files |
| `summary.json` | Every headline number, written by `analyze.py` |
| `tables/*.csv` | The tables behind the paper, written by `analyze.py` |
| `figures/*.png` | Figures 1 to 7 of the paper, written by `analyze.py` (300 dpi) |

To regenerate `summary.json`, `tables/` and `figures/`, run this from the repository root:

```bash
python results/analysis/analyze.py --run results/2026-09-01_final --videos media/videos --out results
```

## Which numbers to use

The run was scored twice.

- **Live scoring** happened during the run and is stored in `results.csv` and `results.json`. Its accuracy counted the disrupt and recover phases together, which is more lenient than the pre-registered definition.
- **Replay scoring** (`ablation_scores.json`) ran every recorded frame through the pipeline again, once with the VLM and once without, and counted the disrupt phase only.

Use the replay for any with/without-VLM comparison and for accuracy. The two differ slightly on false belief (193 frames live vs. 196 replayed for the full system) because the replay reads compressed JPEGs, which moves a few detector scores across the threshold. See the paper, Sections 4.4 and 4.5, and `experiments/PROTOCOL.md`.

## Data dictionary

### `2026-09-01_final/results.csv` (one row per trial, live scoring)

| Column | Meaning |
|---|---|
| `trial_no` | 1 to 48, in the order run |
| `scenario`, `rep`, `clutter` | Which condition (`clutter` is `none` or `high`) |
| `verdict` | PASS / PARTIAL / FAIL from the live scorer |
| `during_disruption_acc` | Live accuracy, disrupt and recover phases, edge frames excluded |
| `false_belief_frames` | Scored frames where the system believed the cup was present and it was absent |
| `lost_while_present` | Scored frames where the cup was present and the system did not believe it. Counts every scored phase, baseline included |
| `spurious_reacquisition` | True if the trial ended CONFIRMED on a cup that was gone |
| `scored_frames`, `edge_frames`, `total_frames` | Frame counts. Edge frames (within 1 s of a phase change) were processed but not scored |
| `vlm_calls` | VLM calls during the trial |
| `wall_clock_s` | Seconds the trial took, including operator staging |
| `failure_mode` | Reserved for aborted trials; empty for all 48 |
| `trial_dir`, `first_frame`, `last_frame` | Where the frames were on the lab laptop (the frames themselves are not in this repository) |

### `2026-09-01_final/results.json` (the same 48 trials, plus every frame)

Each trial has the fields above and a `frames` list. Each frame has:

| Field | Meaning |
|---|---|
| `phase` | `baseline`, `disrupt` or `recover` |
| `t` | Seconds since the start of this phase (resets each phase) |
| `present` | Ground truth: was the cup there |
| `status` | Belief state: CONFIRMED, OCCLUDED, DISPLACED, MISSING or AMBIGUOUS |
| `believes` | Whether the system treated the cup as present (true for CONFIRMED and OCCLUDED) |
| `edge` | True if the frame was within 1 s of a phase change and so not scored |
| `detected`, `detector_conf` | Whether YOLO-World returned a "red cup" box, and its confidence |
| `mean_brightness` | Mean pixel brightness of the frame |
| `step_ms` | Time the pipeline took on this frame |
| `vlm_calls` | Running total of VLM calls in this trial (a frame where it goes up is a frame where the VLM ran) |
| `raw`, `annotated` | File names of the saved frames on the lab laptop |

### `2026-09-01_final/ablation_scores.json` (replay scoring)

`per_episode[condition][episode]` holds one record per trial for each condition: `D_full`, `E_no_vlm`, and `D_full(decay=12)`, `D_full(decay=30)`, `D_full(decay=60)`. (`D_full` and `D_full(decay=12)` are the same system.) Each record has:

| Field | Meaning |
|---|---|
| `scenario`, `rep` | Which trial |
| `axis` | Category: control, environment (lighting, camera_pose), occlusion (occlude_hand, occlude_object, removal), unexpected (substitution), identity (identity_swap) |
| `durAcc` | During-disruption accuracy, disrupt phase only |
| `false`, `lost` | False-belief and lost-while-present frames |
| `spurious` | Spurious reacquisition (true/false) |
| `vlm_calls` | VLM calls in the replay |

`corpus` gives the totals (48 episodes, 8,428 frames, 5,699 scored, 691 with the cup absent).

### Other files in `2026-09-01_final/`

- `experiment_config.json`: the pre-registered settings, trial order and clutter definitions.
- `run_log.txt`: timestamped log of the session, including the two restaged trials.
- `final_experiment.py.snapshot`: the exact script that ran.
- `baseline/`: ten bare-rig frames taken before the first trial, with `baseline.json`.
- `trials_sample/`: annotated frames from trials 9, 14, 29, 31, 45 and 46, the trials the paper discusses.
- `README.md`: written by the script at the start of the run.

The full capture (3.1 GB, about 16,900 files) is not on GitHub. It is kept by the author and will be archived with the tagged release.
