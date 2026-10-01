# AVI Research Project — Assignment Status Report (Assignments 1–5)

> Source: Google Doc "AVI_Assignment_Status_Report" · (private Google Drive link removed)
> Created 2026-07-24 · Last modified 2026-07-24 · Imported 2026-09-28 (verbatim)

*Covering Assignments 1–5 from mentor session notes*

# Assignment 1: Figure Out What's Actually Running Your VLM
**(a) Determine exactly how the VLM is being called**
**STATUS: DONE** The VLM is served locally via Ollama on Windows (RTX 3060 laptop GPU). The model is qwen2.5vl:3b, called through the official Python client (ollama.chat()) in vlm_recovery.py — not the CLI, not a raw REST call. Moondream was also pulled and tested as a comparison model.

**(b) Understand the serving-tool landscape and justify the choice**
**STATUS: DONE** Compared Ollama, llama.cpp, and vLLM. Chose Ollama because: (1) it auto-detects and uses NVIDIA CUDA with zero manual build steps on Windows, unlike llama.cpp which requires a CUDA-enabled compile; (2) vLLM is GPU-throughput-oriented for batched/production serving, which is overkill for a single-stream, one-request-at-a-time robotics loop. This matches the real hardware (Windows + RTX 3060), correcting the original assumption of a MacBook + MLX setup, which does not apply here.

**(c) Confirm whether the GPU is actually being used**
**STATUS: DONE** Confirmed directly via nvidia-smi. Idle VRAM ~218MB; during Qwen inference, VRAM jumped to ~4.3GB and GPU utilization spiked to 90–100% repeatedly. This is real, logged evidence — not assumed carryover from the earlier Mac/MLX benchmarks (which only ever covered LLaVA, never Qwen).

# Assignment 2: Benchmark the VLM Properly
**(a) Measure current VLM latency in the actual system**
**STATUS: DONE** Built two scripts: test_vlm.py (single-call timing) and benchmark_vlm.py (multi-trial, multi-config benchmarking with CSV export). Latency was measured across cold-start vs warm state, multiple resolutions, and both models, using live webcam frames through the real capture pipeline.

**(b) Log results in a table**
**STATUS: DONE** Representative results (5 trials/config, model warm, camera opened once and reused):

| Model | Resolution | Avg (s) | Min–Max (s) | Usable output? | Config |
| :- | :- | :- | :- | :- | :- |
| Qwen2.5-VL:3B | 1280x720 | 2.87 | 0.38 – 3.56 | 5/5 | num_ctx default |
| Qwen2.5-VL:3B | 320x240 | 2.88 | 2.86 – 2.91 | 5/5 | num_ctx default |
| Qwen2.5-VL:3B | 640x480 | 3.00 | 2.96 – 3.05 | 5/5 | num_ctx default |
| Qwen2.5-VL:3B | 480x360 | ~1.4 | 1.35 – 1.47 | 5/5 | num_ctx=2048 (fix) |
| Moondream | 1280x720 | 0.59 | 0.49 – 0.93 | 0/5 | fails structured prompt |
| Moondream | 320x240 | 0.47 | 0.42 – 0.60 | 0/5 | fails structured prompt |

**(c) Identify the bottleneck**
**STATUS: DONE** The bottleneck was NOT GPU clock scaling (tested and ruled out via nvidia-smi live monitoring and clock-locking with nvidia-smi -lgc) and NOT image resolution. Using Ollama's own reported timing breakdown (prompt_eval_duration specifically), we found the model's default context handling was reprocessing the prompt/image inefficiently on repeated calls (~3.0s prompt_eval vs ~14ms on the first call). Explicitly setting num_ctx=2048 brought steady-state latency down from ~3.2s to ~1.4s, a ~2.3x improvement, and this is now baked into vlm_recovery.py.

**(d) Reduce image size and re-test**
**STATUS: PARTIAL** *Hypothesis partially overturned — reported honestly rather than confirmed as expected.* Contrary to the original hypothesis, image resolution (320x240 vs 640x480 vs full 1280x720) made almost no measurable difference once the model was warm and num_ctx was fixed. The real fix was the context-handling bug found in (c), not resolution. We did keep a modest downscale (480x360) in the final pipeline as a small additional payload-size reduction, but it is not the primary latency fix.

# Assignment 3: Get the VLM Back Into the Loop
**(a) Reframe when the VLM is allowed to be slow**
**STATUS: DONE** Implemented a TRACKING / REACQUIRE state machine in main.py. OpenCV + proportional controller (controller.py) handles all real-time tracking. The VLM is only invoked after 10 consecutive missed detection frames (MISSES_BEFORE_REACQUIRE), consistent with the plan that recovery, not tracking, is the slow/deliberate path.

**(b) Re-enable the VLM call in the REACQUIRE state**
**STATUS: DONE** The VLM is wired into REACQUIRE via vlm_recovery.get_recovery_action(). It receives a clean (non-overlaid) frame, downscaled, plus a scan-history text summary, and returns one action from the fixed vocabulary (scan_left, scan_right, tilt_up, tilt_down, object_found). The VLM call now runs in a background thread (threading module) so the camera feed keeps updating live instead of freezing during the ~1.4s call.

**(c) Confirm the VLM produces usable recovery actions on real hardware**
**STATUS: DONE** Multiple real, logged end-to-end recovery attempts on actual hardware (Arduino-driven servo arm, Logitech C270 camera). Observed behaviors, as actually logged:
- Clean successes: VLM issues scan_right / scan_left / tilt_up / tilt_down, then correctly reports object_found once the cup re-enters frame, and TRACKING resumes.
- Mechanical-limit handling: when the VLM requested movement past a servo's physical bound (base=180), a bound-check overrides the action rather than grinding the servo.
- Statelessness failure mode: the VLM sometimes repeated the same suggestion because it has no memory between calls. Fixed with (1) a repeat counter on the VLM's raw suggestion and (2) an external spatial-memory system (scan_counts / scan history text fed into the prompt) so the system — not the model — remembers what's already been tried.
- Degenerate output: when the target was genuinely out of frame, Qwen sometimes produced repetitive garbage tokens instead of a valid action. Mitigated via prompt changes, temperature/repeat_penalty tuning, and a more robust parser that searches the response for any valid action word instead of requiring an exact match.

# Assignment 4: Proper AI Use Going Forward
**(a) First drafts of documentation/paper content written by me first**
**STATUS: NOT YET DONE** No final paper or documentation prose has been drafted yet this cycle — all work so far has been code, benchmarking, and debugging. This item should be picked up before the writeup phase.

**(b) Literature review — 2–3 sentences per paper, in my own words, on papers I've actually read**
**STATUS: NOT YET DONE** Not addressed this session. This was explicitly flagged previously as needing a redo (papers cited with an empty results table). Still outstanding.

**(c) Understand AI-assisted code before committing it**
**STATUS: PARTIAL** *Practiced, but not yet a submitted/reviewable artifact.* Throughout this session, every bug (bound-clamping in arm.py, the repeat-detection logic, the threading refactor, the temperature/repeat_penalty tradeoff) was diagnosed by reasoning through root cause with evidence (logs, Ollama's own timing breakdown, nvidia-smi output) rather than pasting in a fix blindly. This reflects the spirit of 4(c) but hasn't been checked by the mentor yet.

# Assignment 5: Keep the Lab Notebook Current
**(a) Log every work session between now and next time**
**STATUS: NOT YET DONE** This session's work has not yet been transcribed into the lab notebook itself. All raw material (console logs, benchmark CSVs, findings) exists and is ready to be written up, but the notebook entry has not been created.

**(b) Log all benchmarking from Assignments 1–3**
**STATUS: NOT YET DONE** The serving setup discovery, latency numbers, the num_ctx bottleneck finding, and the REACQUIRE recovery attempts are all documented in this report and in benchmark_results_*.csv files, but have not yet been copied into the actual lab notebook document.

**(c) Match the specificity of the best existing entries**
**STATUS: NOT YET DONE** Pending — depends on (a) and (b) being written up first. This report is structured to make that transcription straightforward (specific numbers, before/after comparisons, and root causes are already captured above).

**Legend:** Green STATUS = complete this session. Amber STATUS = partially addressed / honest caveat noted. Red STATUS = not yet started — flagged for next session.

# Tab 2
Moond dream is smaller and reduces accuracy
