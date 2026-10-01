# Import decisions log — `import-msi` branch

Autonomous import run, executed 2026-09-27. Originals under
`<HOME>\avi\`, `<HOME>\Documents\Arduino\`, etc. were never modified, moved,
or deleted — every operation below reads from those paths and writes only
into `<HOME>\avi-import\` (a fresh clone). Final result: **413 files, 30 MB**
under `import/msi/`, well under the 150 MB cap, so no drop-lowest-value-first
step was needed.

## Search (Step 1)

- Project root: `<HOME>\avi\` — has its own local git repo. `git remote -v`
  → `origin https://github.com/Alm1973/avi-occlusion-recovery.git`, on
  branch `main` (plus a local `core-review` branch, not relevant here). This
  matches the repo the task said "one may be" — confirmed, not just guessed.
- `runs\2026-09-01_233533\` found exactly where expected, single `runs/`
  directory (no other `runs\*` folders exist anywhere searched). Contains
  all the named files (`results.csv`, `results.json`, `ablation_scores.json`,
  `run_log.txt`, `experiment_config.json`, `final_experiment.py.snapshot`,
  `README.md`, `baseline\`, `trials\` — 48 trial subfolders, 16,867 files,
  3.1 GB).
- August synthetic/live corpus + ablation scripts: these live as loose
  top-level files in `avi\` (e.g. `nano_*.json`, `openvocab_*.json`,
  `decay_sweep.py`, `score_run.py`, `run_study.py`) rather than in their own
  named subfolder — included via the "all non-underscore-prefixed root
  `.py`/`.json`" rule below.
- All named code files found and copied: `final_experiment.py`, `main.py`,
  `tracker.py`, `vlm_recovery.py`, `arm.py`, `llm_backend.py`,
  `controller.py`, `benchmark_vlm.py`, `test_vlm.py`, `findsearch_final.py`
  (+ other `findsearch_*.py`/`live_*.py` tools). **No file literally named
  `detection.py` exists.** Closest matches, all copied since they fell under
  the "all root `.py`" rule anyway: `adaptive_detector.py`,
  `detect_experiment.py`, `detect_methods.py`, `detector_decision.py`,
  `live_detect.py`, `open_vocab_detect.py`, `test_moondream_detect.py`.
- Arduino sketches: searched the whole home directory (excluding AppData,
  venvs, `.git`, model weights) for `*.ino`. Found exactly one:
  `<HOME>\Documents\Arduino\sketch_jul1a\sketch_jul1a.ino`, and confirmed by
  grep that it mentions servo/arm/cup/track — relevant, copied.
- Docs/figures: `report_latency_evolution.png`, `report_accuracy_evolution.png`,
  `live_experiment_chart.png`, 6× `benchmark_results_*.csv`, and the 7 core
  project docs (`DESIGN_NOTES.md`, `LAB_NOTEBOOK.md`, `LIVE_SESSION_PLAN.md`,
  `PROGRESS.md`, `README.md`, `SERVO_CALIBRATION.md`, `WRITEUP.md`) all found
  at `avi\` root and copied. **No file/folder named "Occlusion Recovery
  Research Data Dump" was found** anywhere searched (home dir to depth 8,
  minus excluded paths) — not copied since it doesn't exist under that name;
  if it's a section heading inside one of the copied `.md` files rather than
  a folder, it's already included via those files.
- `<HOME>\Downloads\avi_setup\` reviewed: contains only
  `Miniconda3-latest-Windows-x86_64.exe`, an installer with no project
  content. Not copied.
- `comfyui-setup-notes.md` at `avi\` root reviewed: it's about running
  ComfyUI (image generation) on the same GPU, unrelated to the AVI tracking
  project. Not copied.

## Copy scope (Step 2) — judgment calls

- **Root code/config**: copied all 173 non-underscore-prefixed `.py` files
  and all non-underscore-prefixed `.json` files at `avi\` root (these are
  the project's actual modules plus its many ablation/probe/benchmark
  scripts and their result files — legitimately part of "all code" for a
  research repo, not cleanup candidates).
- **Excluded**: the 80 underscore-prefixed files at `avi\` root
  (`_log_*.txt`, `_*.jpg`, `_*.py` probes) — these are the author's own
  scratch/throwaway-probe naming convention (leading underscore), not
  named in the task's file list, and not distinguishable from one-off
  debugging output. Also excluded: `cycle_runs\` (32 periodic
  health-check logs, not experiment results) and `test-output.png` (a
  generic scratch image, not one of the named report figures).
- **Frame sampling from `runs\2026-09-01_233533\trials\`**: before copying
  any frame content into a *public* repo, checked `avi\.gitignore`, which
  documents a deliberate policy: frames from the project's other (desk/
  bedroom) corpus are never committed because they show a private room, an
  open laptop with readable documents, and the operator. **The same file
  contains a dated exception**: the controlled-rig run — "3.1 GB across
  16,856 files," matching this run's 16,867 files almost exactly — was
  explicitly cleared for publication by the operator on 2026-09-02, because
  it shows only a staged rig (cup, cardboard box, pliers, mugs) against a
  plain background, nothing private. This confirms copying a frame sample
  from `2026-09-01_233533` is consistent with the project's own existing,
  dated decision, not a new privacy call being made here. Two things were
  copied as a result:
  1. `avi\figures\` (13 MB) copied in full — the operator's own curated,
     already-approved evidence set: 48 contact sheets (16 tiles/trial) +
     9 scenario/trial videos + the baseline scan + its README. This is
     higher-value and already vetted, so it's the primary visual artifact.
  2. A hand-picked raw sample (36 annotated JPEGs, 8.0 MB, well under the
     40-image/40 MB cap) from the 6 trials named in the task (9, 14, 29, 31,
     45, 46): 6 evenly-spaced disruption-phase frames per trial, to satisfy
     the literal "annotated frames from trials 9/14/29/31/45/46" request in
     addition to the curated set.
  - Full per-trial frame counts/sizes for what was *not* copied are in
    `FRAMES_NOT_COPIED.txt`.
  - `baseline\` (10 JPEGs, 1.5 MB) copied in full — small, and the task
    explicitly listed it as a folder to retrieve intact.
- **Large files**: nothing over 50 MB was ever a candidate for copying
  (model weights excluded categorically per Step 1's instructions; no
  individual frame JPEG exceeds ~250 KB). See `LARGE_FILES.txt`.

## Secrets / privacy scan (Step 2, hard rule)

Scanned every file staged for commit for: API-key/token/secret/password
patterns, AWS keys, GitHub/Slack tokens, `sk-`-style tokens, PEM/private-key
headers, `.env` filenames, and Tailscale-range IPs (`100.x.x.x`, any range,
not just CGNAT). **Zero hits — nothing excluded or redacted for secrets.**
(First pass of this scan used a shell regex that silently failed to match
backslash-containing patterns; re-ran with fixed-string and corrected
patterns before trusting the "clean" result — see the home-path finding
below, which the first pass also missed for the same reason.)

**Home path redaction**: searched for the literal Windows home directory
path (backslash and forward-slash forms) across every copied text file. Found it in 10
`.py` files — all local paths to the Ollama/llama-server install and GGUF
model files used by various backend-latency/VRAM ablation scripts
(`backend_latency_ab.py`, `cacheram_test.py`, `ctx_test.py`, `erase_test.py`,
`fa_experiment.py`, `llm_backend.py`, `newbin_test.py`, `test_b10310.py`,
`ubatch_test.py`, `vram_test.py`) — plus one instance in this repo's own
`FRAMES_NOT_COPIED.txt`. All 11 files had every occurrence replaced with
`<HOME>` (case-sensitive literal string replacement, verified with a
follow-up scan showing zero remaining matches). No secrets were embedded in
these paths — they were just local install locations — but per the hard
rule they're redacted regardless since the repo is public.

## Git identity

`<HOME>\.gitconfig` has no global `user.name`/`user.email` set. Set both
**locally** in `avi-import\` only (not touching global config): name
"Shaurya (via Claude import)", email `<author email>` (the
address associated with this Claude session). `gh auth status` shows an
existing authenticated session as GitHub user `Alm1973` — matches the repo
owner, so push (Step 4) uses that.

## Post-commit fix: nested `.gitignore` silently dropped 51 files

The first commit copied `avi\.gitignore` into `import/msi/avi/.gitignore`
for provenance (see the frame-privacy note above). Because git applies a
`.gitignore` to everything *under* its own directory regardless of where
in a larger tree that directory sits, this nested copy was still active in
the new repo and silently excluded 51 files that its own rules target
(`*.png`, `*.jpg` under `runs/`, and a `*_token*` pattern) — including all
36 sampled trial frames, the 10 baseline JPEGs, the 3 report PNGs, and
`vision_token_sweep.{py,json}`. Caught by comparing `git ls-tree` (389
tracked) against the file count on disk (414) before push. Force-added all
51 with `git add -f` in a follow-up commit rather than amending the first
one. The original `.gitignore` is left in place as-copied (it's accurate
documentation of the source repo's own policy) — its rules just don't
apply here because everything under `import/msi/avi/` was force-added.

## Not done / out of scope

- Never executed `final_experiment.py`, `main.py`, `controller.py`,
  `arm.py`, `benchmark_vlm.py`, `test_vlm.py`, or any script that opens
  `serial.Serial('COM5', ...)` or a camera device — all read as text only,
  per the hard rule against moving the arm or touching the serial port.
- `ollama show qwen2.5vl:3b` and `ollama list` were run for
  `INTEGRITY_CHECKS.md` check 6/7 — read-only informational queries against
  the already-running local Ollama install, no model was pulled, started,
  or stopped, and neither command touches the camera or Arduino.
- Software versions in `INTEGRITY_CHECKS.md` check 7 are the *current*
  environment (queried 2026-09-27), not a frozen record from the Sept 1 run
  — none of the run's own files logged versions at run time. Flagged as a
  caveat there rather than presented as exact.
