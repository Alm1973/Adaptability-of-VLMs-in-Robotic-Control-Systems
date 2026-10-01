# Project history

These documents show how the project developed from April to September 2026. They are kept as they were written. Where a number here disagrees with the paper, the paper and `results/summary.json` are correct.

## Contents

| Folder or file | Period | What it is |
|---|---|---|
| `early-2026/` | April to July | Docs from the original version of this repository: hardware photos and wiring, the first research question, the system schematic, the spatial-reasoning literature review, and a project notes file |
| `google-docs/early-roadmap-and-open-questions.md` | April | Roadmap and open questions |
| `google-docs/assignment-1-lit-review-questions-hardware.md` | April | First sources, candidate research questions, hardware inventory |
| `google-docs/assignment-2-research-question-physical-ai.md` | April | The first formal research question, the planned runs, and the ordered hypothesis (occlusion easiest, unexpected objects hardest) |
| `google-docs/assignment-2-literature-review.md` | April | Literature review |
| `google-docs/assignment-3-architecture-llava-task-spec.md` | May | Choice of architecture, LLaVA timing, task specification |
| `google-docs/assignment-status-report-jul-24.md` | July | Status report: Qwen2.5-VL through Ollama, the tracking and reacquire loop |
| `google-docs/complete-report-aug-4-8.md` | August | The GPU output bug, the local server, hallucination fixes, per-frame speedup |
| `google-docs/method-metrics-findings.md`, `google-docs/august-ablation-data-V.md` | August | Belief states, the synthetic and live corpora, conditions A to F, and their results (exploratory) |
| `google-docs/final-experiment-prompt.md` | September | Prompt for the coding assistant that set up the final experiment (it asked for four servos; the run used two) |
| `google-docs/final-review-technical.md`, `google-docs/final-review-plain-english.md` | 11 September | Reports on the 1 September run written for my mentor, before the numbers were re-checked |
| `avi-occlusion-recovery-README.md` | September | README of the lab laptop's working repository |

The Google Docs were exported as text. Links to the originals were removed because those files are private.

## Numbers corrected after these were written

1. **Frame rate.** The camera recorded at 10 fps. "About 28 fps" in older reports is the pipeline's processing speed, so every conversion from frames to seconds in them is about 2.8 times too short: 40 frames is about 4 s, not 1.4 s.
2. **Full system vs. no VLM.** Compare the two from the replay only: 196 vs. 297 false-belief frames. Some older tables put the live number for the full system (193) next to the replay number for the no-VLM version.
3. **Trials that differ.** Six (9, 12, 14, 29, 30, 45). The sign test is one-sided p = 0.016, two-sided p = 0.031.
4. **Accuracy and verdicts.** The live scorer counted the disrupt and recover phases. The pre-registered definition, used in the paper, is the disrupt phase only. Under it the 13 PARTIAL trials become FAIL.
5. **Occlusion.** On the bare rig the belief did not survive full occlusion: a covered cup went to MISSING or DISPLACED within about a second. The occlusion passes with clutter mostly came from the detector boxing a different mug.
6. **Setup.** The VLM ran through Hugging Face transformers in 4-bit, not llama-server. The camera captured 1280×720, not 480×360.
