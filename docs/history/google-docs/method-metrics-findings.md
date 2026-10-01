# AVI — Method, Metrics and Findings Explained

> Source: Google Doc · (private Google Drive link removed)
> Created 2026-08-15 · Last modified 2026-08-15 · Imported 2026-09-28 (verbatim)
> Covers the August synthetic + live corpora (pre–Sept 1 final experiment). Companion doc "AVI — Occlusion Recovery Research Data Dump" was NOT found in Drive — likely on the MSI or Mac.

# AVI — how the experiment was run, what it tests, what the numbers mean
Companion to "AVI — Occlusion Recovery Research Data Dump". That document is the raw numbers. This one explains where they came from and how to read them.

## 1. What the system is
A 2-DOF camera arm (pan and tilt) that finds a named object, verifies it is still the right object, and keeps track of it. Three parts do the work:
- **Detector (YOLO-World).** Open-vocabulary — you give it the words "red cup" at runtime, not a fixed class list. Runs on every frame. Returns boxes and confidences.
- **OpenCV geometry.** Cheap per-frame maths: how much changed, where, and whether the change is inside or outside the target's region. Costs about 12 ms.
- **VLM (Qwen2.5-VL-3B, running locally).** Expensive, roughly 180–270 ms per call, so it is invoked only on events — never every frame.

On top of those sits a **belief state machine**. This is the part that makes the project a research question rather than a demo.

## 2. The core idea being tested
A conventional tracker treats every loss of the target identically: it stops seeing the object, so it starts scanning for it. That is wrong in a specific and important way. If a hand is covering the cup, the correct action is to **wait** — the cup has not gone anywhere, and scanning away from it makes things worse.

So the pipeline diagnoses *why* it lost the target before deciding what to do:

| Belief state | What it means | Correct response |
| :- | :- | :- |
| CONFIRMED | Seen and verified as the target | Track it |
| OCCLUDED | Not visible, but something is covering its last known position | Hold still and wait |
| DISPLACED | The whole frame moved — probably the camera, not the world | Re-localise |
| MISSING | Its old position is back to plain background; it left | Search for it |
| AMBIGUOUS | Something is there but identity is disputed | Ask the VLM |

The research question is whether a **locally hosted** VLM (not a cloud API) inside this structure can restore correct belief after a disruption — and critically, **which component actually does the restoring.** That last clause is what forces ablations.

## 3. How a single experiment is structured
Every episode has three phases, and ground truth is known for each because the protocol dictates what happens:
1. **Baseline** — cup in clear view, nothing happening. Target is present.
2. **Disrupt** — the disruption occurs (hand covers it, lights change, object swapped, etc.). Target is present or absent depending on the scenario.
3. **Recover** — the disruption ends. Target is present or absent depending on the scenario.

The pipeline runs continuously across all three. That matters: belief is a *trajectory*, not a per-frame verdict. An early version of the analysis reset the pipeline at each phase boundary, which meant the disrupt phase began with no reference image and every frame came back MISSING. The results looked clean and were wrong.

**Frames near a phase boundary are excluded from scoring.** When a human is halfway through putting their hand over the cup, the target is genuinely neither present nor absent. Those frames are marked at capture time and skipped — about a third of the live corpus. They still pass through the pipeline (so the trajectory stays continuous) but are not graded.

## 4. The two corpora, and why there are two
### Synthetic (36 episodes, 816 frames)
Built by taking a real captured photo and programmatically altering it: pasting rectangles over the cup to simulate occlusion, inpainting it out to simulate removal, multiplying brightness to simulate lighting change, hue-rotating the cup's own pixels to create a "same-class impostor". Ground truth is exact by construction — the script knows what it did.

**Advantage:** cheap, repeatable, large, perfectly labelled.
**Fatal weakness discovered later:** a drawn rectangle is not a hand. See section 7.

### Live (10 episodes, 1,613 frames, 1,099 scored)
A human sits at the desk and performs each disruption while an on-screen prompt tells them what to do and when. The script records every frame with its phase label. Occluders used: hand, book, paper, cardboard box, jacket sleeve.

**Ground truth comes from the protocol, not from reviewing footage afterwards.** This is deliberate — post-hoc labelling has gone wrong twice on this project. The script says "cover the cup now", so during that phase the cup is known to be present-but-hidden.

**Both corpora are graded by the same scoring code**, so their numbers can be placed side by side. A separate scorer for each would have drifted apart within a week and made every cross-corpus comparison unfalsifiable.

## 5. The six conditions (the ablation)
Same corpus, same frames, components switched on and off. The point is to attribute the result to a component rather than to the system as a whole.

| Condition | What is enabled | What it answers |
| :- | :- | :- |
| A_vlm_only | VLM alone | Can the VLM do this by itself? |
| B_detector | Detector + OpenCV | Can classical vision do it alone? |
| C_stateless | Detector + OpenCV + VLM, no memory | How much does the state machine add? |
| D_full | Everything | The complete system |
| **E_no_vlm** | D with the VLM removed | **What does the VLM contribute?** |
| F_probe | D, plus the VLM region probe replacing the decay timer | Does asking "what is covering it?" help? |

**D versus E is the only comparison that isolates the VLM.** An earlier design compared D against C and reported that D won — but C already contains the VLM, so that comparison was measuring the state machine and being credited to the VLM. Fixing this changed the project's headline claim.

## 6. What each metric means
### durAcc — "during-disruption accuracy"
The fraction of frames *within the disruption window* where the system's belief matched reality. Range 0 to 1, higher is better.

**Why it is restricted to the disruption window:** scoring after the disruption ends measures whether the system can see a cup sitting in plain view, which every condition does perfectly. That saturated all conditions at 7/7 and made the study unable to distinguish them. The interesting behaviour only exists while the disruption is happening.

### lost_while_present
Count of frames where the system said "absent" but the target was there.
**Cost:** a wasted search. The robot goes looking for something that never left. Recoverable — it will find it again.

### false_belief
Count of frames where the system said "present" but the target was gone.
**Cost: this is the one that matters operationally.** The robot believes an object is there when it is not, and reaches for empty space. Every design decision in this project treats false belief as worse than a lost frame, and the two are never averaged into a single "accuracy" number — doing so would let a system hide dangerous errors behind harmless ones.

### spurious_reacquisition
Boolean per episode: did it finish CONFIRMED when the target was actually gone? This is terminal false belief — the state the robot would act on.

### valid (search trials only)
Was the cup genuinely out of frame when the trial started? If not, the trial measured *centring*, not *searching*. This flag exists because nine early control trials came back looking like flawless search performance while never searching at all — the camera's field of view was wide enough that the "offset" never removed the cup from view.

### found / centred (search trials)
Recorded separately and never combined. **found** = did it re-acquire the target. **centred** = did it then bring the target to the middle of frame within 90 px. They are kept apart because they fail at different points: at half brightness the system found the cup in every trial and centred it in none. A single "success" number would have hidden that entirely — and the distinction matters, because one is a perception result and the other is a control result.

## 7. The findings, explained
### Finding 1 — The VLM does exactly one thing
D and E produce **identical** results on environment change, occlusion, and unexpected objects. They differ only on identity: false belief 0 versus 24 frames, spurious re-acquisition 0/3 versus 3/3.

**What this means:** the VLM is not what makes the system robust to occlusion — the belief state machine is (52 lost frames without it, 4 with it). The VLM's job is rejecting an impostor: an object of the same class, in the same place, the same size. That defeats every geometric test by construction, so only something that can reason about identity can catch it. Neither component substitutes for the other.

### Finding 2 — The synthetic corpus gives the wrong answer about occlusion
On synthetic data, condition D scores 0.968 and F scores 0.786 — D wins clearly. On real frames, D scores 0.299 and F scores 0.504 — the ordering reverses completely.

**Why:** D holds belief for exactly 12 frames (a parameter called DECAY_FRAMES) and then gives up. Synthetic disruptions are 4, 8, or 14 frames long. Real occlusions run 40–61 frames. **The synthetic corpus's longest disruption was shorter than the parameter it was supposed to be testing**, so the timer never expired and D looked perfect.

This is the project's main methodological result: a simulated disruption that is easier than the real thing does not merely under-estimate difficulty — it can invert which design you choose.

### Finding 3 — The region probe works on hands and almost nothing else
The probe's job is to answer the hardest question in the system: something is covering the target's position — is the target *behind* it (wait) or has it been *replaced* (give up)? From a single frame those are pixel-identical, so the probe asks the VLM "what is the main object here?" and classifies the answer against a list of things that plausibly cover objects.

On real occluders it gets 1 of 5 right. The reason is visible in the answers:

| Real object | VLM called it | In the list? | Result |
| :- | :- | :- | :- |
| a hand | "hands" | yes | durAcc 1.000 |
| a sheet of paper | "document" | no | durAcc 0.000 |
| a cardboard box | "notebook" | no | durAcc 0.129 |
| a jacket sleeve | "socks" | no | durAcc 0.143 |
| a book | "desk lamp", "notebook" | mostly no | durAcc 0.562 |

**What this means:** the list is not too short. Real-world naming is unpredictable — a jacket reads as socks, a box as a notebook. No vocabulary fixed in advance survives that. The fix is not a longer list (the next unlisted noun fails identically) but posing the question as a *classification* rather than a string match.

There is a second-order effect worth understanding: when the probe cannot name the occluder it concludes "replacement" and drops belief **immediately**, whereas the timer it replaced held belief for 12 frames. So on unnameable occluders the probe is *worse* than doing nothing clever. **A wrong answer arrives faster than no answer.**

### Finding 4 — Detection is bounded by size and coverage, nothing else
Nine degradation axes were swept on real frames. Six never break the detector at all, including things one would expect to matter:
- **Darkness** — at 8% of captured brightness (twelve times darker) it still returns 0.735 confidence with the box essentially unmoved.
- **Blur** — 22 px of Gaussian blur, still 0.971.
- **Motion blur** — a 71 px streak, far worse than the arm produces while panning, still detects on every frame.
- **Frame-edge truncation, local shadow, glare** — all no effect.

Only two things bound it: **apparent size** (fine to 0.2× resolution, gone by 0.12×) and **coverage fraction** (holds past 70%, fails between 85% and 100%). Coverage direction is irrelevant — top, bottom and side all fail at the same point, so the detector does not depend on seeing the cup's rim.

**Practical consequence:** the operating envelope is about distance and how much is hidden, not about lighting or focus. Effort spent on low-light robustness would be wasted.

### Finding 5 — Search has a hard structural limit
Every experiment above uses a *static* camera, so they all measure **verification** ("is it still there, is it still the right one?"). Search — the arm actually moving to find the cup — was only tested at the end, and immediately exposed a limit that no static test could see:

The search policy takes 12° steps and allows 2 steps per direction, so it can travel **24° maximum** before declaring that direction exhausted. A target lost more than 24° away cannot be re-acquired — the policy gives up by construction, not because it failed to perceive anything.

Also: the synthetic study found random search beat VLM-guided search (0.771 versus 0.429). That was measured over a much smaller space and does not transfer — in live trials random search burned all 24 permitted moves on a plainly visible cup.

### Finding 6 — The bottleneck is not where it looks
The VLM identity check is the most expensive single call in the system at 181 ms, and it accounts for **3.4%** of runtime — because it fires roughly once per episode. Meanwhile the detector (13.9 ms, but every frame) and the region probe together account for 82%.

Three optimisations were tried and all rejected on measurement: half-precision inference gave 1.03× with bit-identical outputs (the flag was probably ignored), reducing input size 640→448 gave 1.04× despite 2.3× less compute (so the detector is not compute-bound — its 12 ms is fixed overhead), and cutting the probe's token budget did nothing because generation stops at end-of-sequence long before the cap.

The real ceiling is the camera: it delivers 10.1 fps while the pipeline can process 29.4. **Capture, not inference, limits throughput.**

## 8. How to read the caveats
Several results in the data dump are marked void or provisional. What those labels mean:

| Label | Meaning |
| :- | :- |
| **Void** | Collected under conditions that make it uninterpretable — do not cite. The brightness search sweep is void because its control failed and it predates a bug fix. |
| **Provisional** | Real measurement, but with a known deviation. The search trials raised the search budget from 2 to 6 steps per direction so success was possible at all — so those numbers do not describe shipped behaviour. |
| **Open gap** | Something we tried to measure and failed to. The live corpus has **zero valid identity episodes** after three capture attempts, so Finding 1 — the headline claim about the VLM — rests on synthetic data alone. |

## 9. Sample sizes and what they support
The live corpus is n = 1 per scenario; the search trials are n = 2–5 per condition. That supports one-directional claims — "the probe fails on paper", "search cannot exceed 24°" — because those are failures visible at n = 1. It does **not** support ranking conditions separated by small margins. A two-frame difference is one operator's hand movement.

Everything was collected on one desk, with one operator, one camera, and one target object.

## 10. Why the bug list is in the report
Nine measurement bugs were found and fixed, and they are documented alongside the results rather than quietly corrected. Each one produced a confident, plausible, wrong answer first — for example, substring keyword matching classified "notebook", "handbag" and "armchair" as occluders, which would hold belief on a target that is gone. Most were caught only because two independent measurements disagreed.

The pattern worth carrying forward: **a result that looks clean is not evidence that it is correct.** The nine control trials that scored 9/9 were measuring the wrong thing entirely, and the only clue was a column of zeros in a field nobody was looking at.
