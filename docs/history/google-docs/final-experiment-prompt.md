# AVI — Final Experiment Run (Claude Code prompt)

> Source: Google Doc "AVI_final_experiment_prompt.md" · (private Google Drive link removed)
> Created 2026-09-02 · Imported 2026-09-28 (verbatim)
> Note: this prompt asked for four servos, but the Sept 1 final experiment as reported ran on a 2-DOF (pan/tilt) arm — see final-review-technical.md provenance.

# AVI — Final Experiment Run

## Context updates since our last session
- **Network:** I switched the machine from Wi-Fi to **Ethernet**. Latency and dropped-frame issues from the last runs should be gone. Don't design around network flakiness anymore — but do log timing so we can confirm it empirically.
- **Physical setup:** I've built a **controlled environment** for the bot — fixed staging area, consistent lighting, marked object positions. It's stable and repeatable now, which means we can finally run a real experiment instead of ad-hoc trials.
- **Arm:** The bot has **four servos**. I want all four engaged and coordinated — full range, smooth interpolated motion, not the one-axis-at-a-time jerky movement from before.

---
## Step 0 — Orient yourself before touching anything
Read the project first. Don't assume anything from memory:
1. Read arm.py, tracker.py, vlm_recovery.py, and main.py.
2. **Enumerate the actual servo configuration** — channel/pin per servo, what joint each one drives, the safe min/max angle for each, current home position, and how motion commands are issued.
3. Report that back to me in a short table before you run anything.
4. Confirm the VLM is loaded correctly (GPU-first, all layers offloaded) and that the degenerate-output retry guard is active.
5. Tell me what you understand the experiment to be testing, based on the project's own history and code. Don't ask me to restate it — you already have the context. State your understanding, and I'll correct you if you're off.

---
## Step 1 — Baseline environment scan (do this)
Go ahead and drive the bot to scan the controlled environment before we run trials. I want a real baseline, not an assumption:
- Sweep the workspace using the servos and capture the scene from several viewpoints.
- Run the VLM on the baseline captures and record what it sees: objects, positions, labels, confidence.
- Record ambient conditions you can measure — frame brightness/exposure, capture latency, inference latency per frame.
- Save this as the **baseline** the experimental trials get compared against. Every later observation should be diffable against it.
- Show me the baseline summary and flag anything that looks off *before* we proceed. If the scene doesn't match what the experiment needs, say so and stop.

---
## Step 2 — Servo motion requirements
All four servos, working together:
- **Engage all four** on every movement sequence — no single-joint moves unless the geometry genuinely requires it.
- **Smooth motion:** interpolate between waypoints (ease-in/ease-out, small angular steps at a fixed tick rate) rather than commanding target angles directly. No snapping.
- **Coordinated:** joints move simultaneously toward a pose, not sequentially.
- **Full usable range:** use the actual limits you discovered in Step 0, with a safety margin. Don't artificially restrict the workspace, but never command past a safe limit.
- Add a move_smooth(pose, duration) style helper if one doesn't exist, and route every motion through it.
- Home the arm safely at the start and end of every trial so each trial begins from an identical state.

---
## Step 3 — Design the experiment properly
This is the final run, so it needs to hold up as an actual experiment, not a demo. Build it with:
- **A stated hypothesis and the variables** — what's independent, what's dependent, what's held constant. Write these into the script header and the log.
- **A control condition** and clearly defined **experimental conditions** (the disruptions/variations being introduced).
- **Repeated trials per condition** — enough replicates to say something, not one-shot anecdotes. Pick a defensible N and justify it.
- **Randomized or counterbalanced trial order** where it matters, so ordering effects don't confound results.
- **Pre-registered success criteria** — define what counts as a success, a partial, and a failure *before* the run, in code, so scoring isn't post-hoc.
- **Per-trial metrics:** outcome, VLM output, confidence, servo commands issued, wall-clock timing at each stage, retry/recovery attempts, and failure mode if it failed.
- **Structured logging** — one row per trial in a CSV (plus a JSON with the full detail), so this can go straight into analysis.
- **Aborts and safety:** any unsafe pose, VLM degenerate output, or hardware timeout halts that trial cleanly, logs the reason, and re-homes — it never crashes mid-run or leaves the arm loaded.

---
## Step 4 — Make it runnable by me, step-gated
I'm going to run this myself and physically set up each condition. Design for that:
- I run one command to start it. Everything else is driven by the script.
- **The script tells me exactly what to do.** Before each trial it prints a clear, numbered instruction — what to place, where, in what orientation, what to change — then waits for me to press ENTER to confirm.
- Instructions must be unambiguous and reference the marked positions in my controlled environment. "Place the target object at the marked center position, upright" — not "set up the scene."
- After each trial it prints the result so I can see it working: what the VLM saw, what the arm did, pass/fail.
- Let me **abort, skip, or repeat** a trial from the prompt if I fumble the physical setup.
- Show progress: trial X of N, condition name, elapsed time.
- If it crashes or I stop it, it must be **resumable** — don't make me redo completed trials.
- Print a summary at the end: results per condition, aggregate metrics, and where the raw data landed.

---
## Step 5 — Image capture and folder structure
Every image gets saved to disk so I can cross-reference them by hand afterward:
- Create a run folder: runs/<YYYY-MM-DD_HHMMSS>/
- Inside it:
  - baseline/ — the Step 1 scan images
  - trials/trial_<NNN>_<condition>/ — every frame captured during that trial
  - results.csv, results.json, run_log.txt, experiment_config.json
- **Filenames must be self-describing** — trial number, condition, stage (pre-action / post-action / recovery-attempt-N), and timestamp — so I can tell what an image is from its name alone with nothing else open.
- Every image path is written into the CSV row for its trial, so a row and its images are always linked both directions.
- Save the raw frame **and**, where useful, an annotated copy with detections/bounding boxes drawn on it — keep both, never overwrite the raw.
- Nothing gets deleted or overwritten between runs. Each run is its own folder.
- Write a short README.md into the run folder explaining the structure, the hypothesis, and the conditions, so the folder is interpretable months from now.

---
## What I want back from you
1. The servo table and project-state report from Step 0.
2. Your statement of what this experiment is testing, with the hypothesis and variables written out.
3. The experiment design — conditions, N, trial order, success criteria.
4. Then build it, run the baseline scan, and hand me the single command to start the trials.

Ask me before you run anything that physically moves the arm outside the baseline scan. Everything else, go ahead.
