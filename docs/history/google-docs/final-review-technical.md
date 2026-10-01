# AVI — Final Review (TECHNICAL version, for mentor)

> Source: Google Doc · (private Google Drive link removed)
> Created 2026-09-11 · Last modified 2026-09-11 · Imported 2026-09-28 (verbatim; no comments on doc)
> Note: the doc begins at section 3; sections 1–2 are in "Final Review" (final-review.md).

## 3. Headline numbers
| Quantity | Value |
| :- | :- |
| Trials | 48 (8 scenarios × 3 reps × 2 clutter levels) |
| Frames captured | 8,428 — 5,699 scored, 2,729 excluded as edge frames |
| Absent-truth frames | 691 |
| Verdicts | 25 PASS · 13 PARTIAL · 10 FAIL |
| Wall clock | 44.7 min of trials, 63 min including staging |
| VLM invocation rate | 135 calls over 8,428 frames = **1.6% of frames** |
| Step latency, no VLM | median 28.8 ms, p95 48.5 ms |
| Step latency, VLM frame | median 333.6 ms, p95 465.6 ms |

## 4. Method
**Pre-registered before the run**, written into experiment_config.json and the scoring code:
- **Primary DV:** false_belief_frames — asserting the target is present when it is not. Never folded into accuracy, because it is the safety-relevant direction.
- **Secondary:** during-disruption accuracy, lost_while_present, spurious reacquisition.
- **Verdicts:** PASS = acc ≥ 0.7 and zero false belief. PARTIAL = acc ≥ 0.4 and zero false belief. FAIL = anything else, or *any* false belief at all.
- **Held constant:** camera pose (re-homed to 90/90 each trial), lighting, target position, occluder set, phase durations, target string.
- **Ground truth** recorded at capture time from the staging instruction — never assigned by reviewing footage afterwards.
- **Edge frames:** frames within 1.0 s of a phase boundary are stepped through the pipeline so the belief trajectory stays continuous, but excluded from grading. 4.0 s of each 6.0 s phase is scoreable.
- **Order** counterbalanced by block, seed 20260901, so scenario order and clutter order both vary across reps.

Three components, deliberately separated so the ablation can attribute cause: a detector (YOLO-World, every frame), OpenCV geometry (on loss only), and the VLM (event-gated, 1.6% of frames).

### The eight scenarios
| Scenario | Axis | Disrupt phase | Recover phase | Ends |
| :- | :- | :- | :- | :- |
| control | control | nothing changes | nothing changes | present |
| occlude_hand | occlusion | hand fully covers cup | hand removed | present |
| occlude_object | occlusion | cardboard box hides cup | box removed | present |
| removal | occlusion | hand covers cup | cup taken away under cover | **absent** |
| substitution | unexpected | hand covers cup | swapped for pink pliers | **absent** |
| identity_swap | identity | hand covers cup | swapped for different-coloured cup | **absent** |
| lighting | environment | rig lamp off / dimmed | lighting restored | present |
| camera_pose | environment | arm pans itself 12° | arm pans back | present |

**Clutter — none:** bare rig, red cup alone. **high:** six to eight objects in frame including at least two other cups or mugs of different colours, crowded around the marked position without covering the red cup. Same object set all session.

## 48 trials
| # | Scenario | Rep | Clutter | Verdict | Acc | False belief | Lost | Spurious | Scored | Edge | Total | VLM calls | Sec | Det FP on absent | E_no_vlm false |
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | camera_pose | 1 | none | **PASS** | 1.0 | 0 | 0 | False | 118 | 54 | 172 | 1 | 83 | — | 0 |
| 2 | removal | 1 | none | **PARTIAL** | 0.532 | 0 | 37 | False | 119 | 55 | 174 | 1 | 121 | 0% | 0 |
| 3 | occlude_hand | 1 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 54 | 174 | 2 | 39 | — | 0 |
| 4 | occlude_object | 1 | none | **PARTIAL** | 0.5 | 0 | 40 | False | 120 | 57 | 177 | 2 | 56 | — | 0 |
| 5 | identity_swap | 1 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 57 | 177 | 3 | 75 | 0% | 0 |
| 6 | substitution | 1 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 57 | 177 | 1 | 58 | 0% | 0 |
| 7 | lighting | 1 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 57 | 177 | 1 | 84 | — | 0 |
| 8 | control | 1 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 36 | — | 0 |
| 9 | identity_swap | 1 | high | **PARTIAL** | 0.5 | 0 | 35 | False | 110 | 52 | 162 | 16 | 78 | 20% | 26 |
| 10 | occlude_hand | 1 | high | **PASS** | 0.904 | 0 | 7 | False | 113 | 54 | 167 | 11 | 41 | — | 0 |
| 11 | control | 1 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 56 | 176 | 1 | 37 | — | 0 |
| 12 | substitution | 1 | high | **FAIL** | 0.506 | 6 | 32 | False | 117 | 52 | 169 | 5 | 48 | 8% | 9 |
| 13 | camera_pose | 1 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 50 | — | 0 |
| 14 | removal | 1 | high | **PARTIAL** | 0.639 | 0 | 26 | False | 112 | 53 | 165 | 14 | 39 | 100% | 37 |
| 15 | lighting | 1 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 124 | — | 0 |
| 16 | occlude_object | 1 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 100 | — | 0 |
| 17 | occlude_hand | 2 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 56 | 176 | 2 | 37 | — | 0 |
| 18 | control | 2 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 59 | 179 | 1 | 26 | — | 0 |
| 19 | lighting | 2 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 160 | — | 0 |
| 20 | identity_swap | 2 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 57 | 177 | 2 | 74 | 0% | 0 |
| 21 | occlude_object | 2 | none | **PASS** | 1.0 | 0 | 34 | False | 114 | 56 | 170 | 9 | 38 | — | 0 |
| 22 | removal | 2 | none | **FAIL** | 0.487 | 3 | 37 | False | 118 | 57 | 175 | 2 | 38 | 3% | 3 |
| 23 | camera_pose | 2 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 60 | 180 | 1 | 64 | — | 0 |
| 24 | substitution | 2 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 58 | 178 | 1 | 48 | 0% | 0 |
| 25 | camera_pose | 2 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 60 | 180 | 1 | 66 | — | 0 |
| 26 | lighting | 2 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 59 | 179 | 1 | 42 | — | 0 |
| 27 | occlude_object | 2 | high | **PASS** | 0.987 | 0 | 1 | False | 117 | 53 | 170 | 4 | 62 | — | 0 |
| 28 | control | 2 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 59 | 179 | 1 | 31 | — | 0 |
| 29 | removal | 2 | high | **FAIL** | 0.789 | 16 | 0 | False | 116 | 56 | 172 | 5 | 52 | 72% | 36 |
| 30 | identity_swap | 2 | high | **FAIL** | 0.703 | 18 | 4 | False | 114 | 57 | 171 | 9 | 62 | 39% | 22 |
| 31 | substitution | 2 | high | **FAIL** | 0.5 | 40 | 0 | True | 120 | 57 | 177 | 1 | 70 | 100% | 40 |
| 32 | occlude_hand | 2 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 45 | — | 0 |
| 33 | occlude_hand | 3 | none | **PARTIAL** | 0.525 | 0 | 38 | False | 120 | 56 | 176 | 2 | 48 | — | 0 |
| 34 | camera_pose | 3 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 61 | 181 | 1 | 39 | — | 0 |
| 35 | occlude_object | 3 | none | **PARTIAL** | 0.5 | 0 | 40 | False | 120 | 56 | 176 | 2 | 46 | — | 0 |
| 36 | removal | 3 | none | **FAIL** | 0.506 | 1 | 38 | False | 119 | 58 | 177 | 2 | 36 | 0% | 2 |
| 37 | identity_swap | 3 | none | **PARTIAL** | 0.519 | 0 | 38 | False | 119 | 53 | 172 | 4 | 41 | 0% | 0 |
| 38 | control | 3 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 41 | — | 0 |
| 39 | substitution | 3 | none | **FAIL** | 0.325 | 14 | 38 | False | 117 | 57 | 174 | 2 | 60 | 3% | 14 |
| 40 | lighting | 3 | none | **PASS** | 1.0 | 0 | 0 | False | 120 | 59 | 179 | 1 | 42 | — | 0 |
| 41 | control | 3 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 59 | 179 | 1 | 42 | — | 0 |
| 42 | camera_pose | 3 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 60 | 180 | 1 | 38 | — | 0 |
| 43 | lighting | 3 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 62 | — | 0 |
| 44 | occlude_object | 3 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 52 | 172 | 3 | 35 | — | 0 |
| 45 | identity_swap | 3 | high | **FAIL** | 0.303 | 15 | 38 | False | 116 | 57 | 173 | 6 | 52 | 11% | 28 |
| 46 | removal | 3 | high | **FAIL** | 0.5 | 40 | 0 | True | 120 | 60 | 180 | 1 | 41 | 100% | 40 |
| 47 | substitution | 3 | high | **FAIL** | 0.5 | 40 | 0 | False | 120 | 57 | 177 | 2 | 36 | 100% | 40 |
| 48 | occlude_hand | 3 | high | **PASS** | 1.0 | 0 | 0 | False | 120 | 58 | 178 | 1 | 44 | — | 0 |

## 6. Ablation — what the VLM is actually doing
Every condition replays the *same* 8,428 captured frames through a modified pipeline and is graded by the *same* scorer. Nothing is re-staged, so D vs E isolates exactly one component: the VLM.

| Condition | False belief | Lost while present | Mean acc | Spurious reacq. |
| :- | :- | :- | :- | :- |
| **D_full** (VLM on, decay 12) | **196** | 711 | 0.640 | 2 |
| **E_no_vlm** (VLM removed) | **297** | 657 | 0.653 | 11 |
| D_full, decay = 30 | 229 | 478 | 0.762 | 2 |
| D_full, decay = 60 | 235 | 232 | 0.890 | 2 |

### 6a. By disruption axis
| Axis | n | D false | E false | D acc | E acc |
| :- | :- | :- | :- | :- | :- |
| control | 6 | 0 | 0 | 1.000 | 1.000 |
| environment | 12 | 0 | 0 | 1.000 | 1.000 |
| occlusion | 18 | 60 | 118 | 0.516 | 0.520 |
| identity | 6 | 36 | 76 | 0.183 | 0.301 |
| unexpected | 6 | 100 | 103 | 0.392 | 0.358 |

**Read this carefully.** On the identity axis the VLM halves false belief (76 → 36) while accuracy *falls* (0.301 → 0.183). It is not verifying identity; it is suppressing belief indiscriminately and happening to be right more often than wrong. Any writeup that reports the 76 → 36 without the accuracy drop is misleading, and a reviewer will find it.

### 6b. The six trials that differ — and why they are the whole result
| Trial | Clutter | D_full | E_no_vlm | Frames saved |
| :- | :- | :- | :- | :- |
| 009 identity_swap rep1 | high | 0 | 26 | +26 |
| 012 substitution rep1 | high | 6 | 9 | +3 |
| 014 removal rep1 | high | 0 | 37 | +37 |
| 029 removal rep2 | high | 15 | 36 | +21 |
| 030 identity_swap rep2 | high | 19 | 22 | +3 |
| 045 identity_swap rep3 | high | 17 | 28 | +11 |

**42 of 48 trials show zero difference.** Two trials carry 63 of the 101 frames saved (62%). So 101 is not an effect size — it is six episodes, and mostly two of them. The defensible statistical claim is directional: 6 of 6 differing trials favour the VLM, sign test **p = 0.016**. That says the VLM never made things worse in this corpus. It does not establish a magnitude.

**But all six are high-clutter.** That is not a coincidence, and it is the finding.

## 7. The actual finding: the VLM is a detector-false-positive filter
Splitting the ablation by clutter instead of by axis:

| Clutter | n | D_full false | E_no_vlm false | Δ |
| :- | :- | :- | :- | :- |
| none (bare rig) | 24 | 19 | 19 | **0** |
| high (distractor cups) | 24 | 177 | 278 | **+101** |

The reason is the detector. YOLO-World was asked for "red cup". On frames where the red cup was *genuinely gone*:

| Condition | Absent frames | Detector fired anyway | False-positive rate |
| :- | :- | :- | :- |
| clutter = none | 354 | 2 | **0.6%** |
| clutter = high | 337 | 211 | **62.6%** |

| Scenario × clutter | Detector FP on absent frames |
| :- | :- |
| removal / high | 103 / 113 = **91.2%** |
| substitution / high | 83 / 117 = **70.9%** |
| identity_swap / high | 25 / 107 = 23.4% |
| removal / none | 1 / 117 = 0.9% |
| substitution / none | 1 / 117 = 0.9% |
| identity_swap / none | 0 / 120 = 0.0% |

Confidence on those false detections: median 0.383, range 0.174–0.820. The detector is not marginally unsure — it is confidently calling a blue mug a red cup.

**So the causal chain is:** distractor cups in frame → open-vocabulary detector fires on the wrong cup → pipeline would hold a false belief → the VLM is gated on, looks at the crop, and refuses. That is a coherent, mechanistic, defensible claim, and it is *better* than the vague "VLM improves recovery" framing, because it says exactly when the VLM is worth its 333 ms and when it is dead weight.

It also reframes the negative result usefully: the VLM contributed nothing on the bare rig not because it is useless, but because there was no error available for it to catch.

## 8. What the data does *not* support
1. **"The VLM improves occlusion recovery by 34%."** No. 42/48 trials are identical. Two trials carry 62% of the difference. Report the sign test or report nothing.
2. **"The system recovers from occlusion."** The two scenarios the research question is literally named after — occlude_hand and occlude_object — show **zero** false belief on both sides of the ablation. The VLM contributes nothing there. Belief survives hand occlusion because of the 12-frame decay timer, which is a counter, not reasoning.
3. **"The VLM verifies identity."** It reduced false belief on the identity axis while accuracy fell. That is suppression, not verification.
4. **Clutter as a clean independent variable.** Clutter is confounded with detector reliability by construction — that *is* its mechanism of action. Worth stating outright rather than letting a reviewer discover it.
5. **n = 48 for a per-axis claim.** Six trials per scenario, three per scenario×clutter cell. The axis-level table is descriptive, not inferential.

## 9. Secondary findings worth keeping
- **Event-gating works.** 1.6% VLM invocation rate. Median frame cost 28.8 ms; the pipeline only pays 333 ms when something has actually gone wrong. This is the strongest engineering result in the project and it is fully supported.
- **Spurious reacquisition 11 → 2** with the VLM on. Independent of the false-belief count and pointing the same direction.
- **The decay timer is a real tunable.** Going 12 → 60 frames buys +0.25 accuracy for +39 false-belief frames. Whether that trade is good depends entirely on whether the robot is about to grasp something.
- **Environment change is solved.** 12/12 perfect on lighting and camera_pose, both sides of the ablation. Self-motion notification means commanded camera movement is not treated as a disruption at all.

## 10. What to do next
**Highest value, ~35 minutes:**
`python final_experiment.py --reps 8 --clutter none,high --scenarios removal,substitution,identity_swap`
This triples n on exactly the three scenarios where anything happens, and turns "6 of 6, p = 0.016" into a per-scenario effect you can defend. It is the difference between a directional claim and a result.

**Second:** log the detector's predicted class and box alongside confidence, so the false-positive claim can be shown rather than inferred — a frame where the detector boxed a blue mug and labelled it "red cup" is the single most persuasive figure this project could produce.

**Third:** re-run the ablation with the detector's confidence threshold raised. If raising it removes the false positives, the honest conclusion is that a threshold tweak buys most of what the VLM buys for 0 ms — and the VLM's case then has to rest on the cases a threshold cannot fix.

## 11. Provenance
- Run directory: runs/2026-09-01_233533/ — every raw and annotated frame retained, never overwritten.
- results.csv (48 rows) · results.json (4.5 MB, full per-frame belief trajectory) · ablation_scores.json · run_log.txt · experiment_config.json · final_experiment.py.snapshot (the exact code that ran).
- Baseline scan before trials: cup detected in 10/10 frames, brightness 144.
- Two trials were re-staged on operator request before scoring (trial 3 occlude_hand, trial 6 substitution); both re-stagings are in the log and the discarded attempts were not scored.
- Hardware: Arduino Uno + PCA9685, 2-DOF arm homed to 90/90; Logitech C270 at 480×360; VLM on a 6 GB GPU via llama-server.
