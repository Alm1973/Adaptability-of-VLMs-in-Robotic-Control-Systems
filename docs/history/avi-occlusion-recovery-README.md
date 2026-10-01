# AVI — occlusion recovery for robotic object verification

A 2-DOF camera arm that finds a named object, verifies it is still that
object, and keeps track of it through disruptions.

**Research question**

> Can a locally hosted vision-language model, embedded in a reasoning pipeline
> that uses OpenCV for spatial reasoning, recover from disruptions to object
> verification caused by occlusion, environment change, and unexpected objects?

Sharpened, because "is it possible" is satisfied by one lucky episode:
**under which disruption types does the pipeline restore correct belief, how
reliably, and which component is responsible?** The last clause is what forces
ablations — without them you cannot tell whether the VLM, OpenCV, or the
pipeline structure did the work.

---

## Headline results

| finding | evidence |
|---|---|
| The VLM reduces false belief and never increases it | 6 of 6 differing trials favour it, 0 against; sign test p = 0.016 |
| Its contribution is invisible in accuracy | identity: removing it raises accuracy 0.183 to 0.301 while doubling false belief 36 to 76 |
| On environment change the VLM changes nothing | 18 episodes, 18 calls, identical in every column |
| Occlusion never produces false belief | 12 of 12 trials, both conditions, every timer value |
| The VLM cannot help where it is never asked | substitution 103 to 100; 2.0 calls/episode against 6.5 on identity |
| The decay timer trades accuracy against safety, unevenly | 12 to 60 costs +1 false belief on occlusion, +34 on unexpected |
| Occlusion recovery is the **belief state machine**, not the VLM | 52 lost frames without state, 4 with it |
| The synthetic corpus **inverts** the occlusion result | D 0.968 synthetic to 0.299 live; F 0.786 to 0.504 |
| F_probe's occlusion win is a **string match**, not reasoning | delete 5 words from the vocabulary and 1.000 to 0.000 |
| A **timer of 60** matches a *perfect* probe | oracle vocab, oracle probe and timer=60 all give 0.697 / 58 / 5 |
| The region probe works on hands and almost nothing else | 1 of 5 real occluders named correctly |
| `search_direction()` has **never returned a direction** | asks a 4-way question of a yes/no comparator, returns `None` every frame |
| Detection is bounded by **apparent size and coverage only** | 6 of 9 degradation axes never break it; dies below 0.12x resolution and above 85% coverage |
| Search cannot re-acquire a target lost >24 degrees away | `SEARCH_STEP` 12 x `MAX_PER_DIRECTION` 2 |
| The bottleneck is the camera, not the model | 10.1 fps capture vs 29.4 fps pipeline capacity |

**What the numbers do not support.** 42 of the 48 controlled-rig trials show
no difference between the two conditions, and two trials carry 62% of the
total gap, so the direction of the VLM effect is defensible and its magnitude
is not. Detection rate also varies 32% to 100% inside a single scenario name,
which is an uncontrolled staging variable that predicts false belief better
than the condition does. Both are set out in `WRITEUP.md` section 5d.

**Why the corpus inverted the answer:** condition D gives up after exactly
`DECAY_FRAMES` = 12. Synthetic disruptions are 4/8/14 frames; real occlusions
run 40–61. The corpus's longest disruption was shorter than the parameter it
existed to test.

---

## The pipeline

Belief is explicit and diagnosed, not inferred from a miss counter. A
conventional tracker scans whenever it loses the target — which is exactly
wrong if a hand is covering it.

| state | meaning | correct response |
|---|---|---|
| `CONFIRMED` | seen and verified | track |
| `OCCLUDED` | hidden, something overlaps its last position | hold and wait |
| `DISPLACED` | whole frame moved — probably the camera | re-localise |
| `MISSING` | region reverted to background | search |
| `AMBIGUOUS` | something there, identity disputed | ask the VLM |

**Tier split** — OpenCV owns geometry, the VLM owns semantics:

| tier | cost | frequency |
|---|---|---|
| detector (YOLO-World) | 13.9 ms | every frame |
| OpenCV diagnosis | 11.7 ms | on loss |
| VLM (Qwen2.5-VL-3B) | 181–271 ms | **on event only** |

Gating the VLM is what lets an expensive verifier coexist with a real-time
loop: it is the most expensive single call in the system and 3.4% of runtime.

---

## Layout

| file | purpose |
|---|---|
| `recovery_pipeline.py` | belief state machine + OpenCV diagnosis |
| `run_study.py` | six-condition ablation, synthetic corpus |
| `live_bench.py` | same six conditions scored on real frames |
| `disruption_bench.py` | synthetic corpus builder |
| `live_disruption.py` | live capture protocol (human performs disruptions) |
| `nano.py`, `search_trials.py` | live search trials — **the arm moves** |
| `probe_real.py`, `probe_generalise.py` | region-probe naming tests |
| `degrade_bench.py`, `search_challenge.py` | degradation envelope |
| `profile_pipeline.py`, `speed_opts*.py` | timing, and speedups rejected on evidence |
| `find_live.py` | the shipping tracker |
| `assignment2.py` | is F_probe's occlusion win reasoning or a string match? |
| `assignment3a.py` | VLM contribution per axis, with call cost and belief diff |
| `assignment3b_offline.py`, `search_prompt_sweep.py` | directional-search audit |
| `PROGRESS.md` | full append-only research log |
| `LAB_NOTEBOOK.md` | structured notebook entries |
| `LIVE_SESSION_PLAN.md` | what still needs hardware, and what is blocked |

Live and synthetic corpora are graded by **one shared scorer**
(`score_episode` in `run_study.py`), so their numbers are directly comparable.
A second scorer would have drifted within a week and made every cross-corpus
claim unfalsifiable.

---

## Running it

```bash
python run_study.py                    # synthetic ablation, all six conditions
python live_bench.py D_full F_probe    # score chosen conditions on real frames
python degrade_bench.py                # degradation envelope, no hardware
python probe_real.py                   # what the VLM calls each real occluder
```

**These move the arm — a human must be present and watching:**

```bash
nano.bat list          # show conditions
nano.bat clear 3       # control, 3 trials
nano.bat all 3         # every condition, prompts between each
```

Hardware: Arduino Uno (COM5) + Logitech C270 + RTX 3060 Laptop 6 GB.
Qwen2.5-VL-3B runs 4-bit in-process via `transformers` — there is no server to
start.

---

## Ground truth

Recorded by the capture protocol **at capture time**, never assigned by
reviewing footage afterwards. The script tells the operator what to do and
when, so "present" is known by construction.

Frames near a phase boundary — where a hand is halfway onto the cup and the
truth is genuinely undefined — are marked `edge`, passed through the pipeline
so the belief trajectory stays continuous, and excluded from grading. That is
about a third of the live corpus.

---

## Known limits

- **Zero valid identity episodes on real hardware** after three capture
  attempts. The headline VLM result rests on synthetic data alone.
- **The VLM has never steered a search.** `search_direction()` asks a four-way
  question of a yes/no logit comparator, so it returns `None` on every frame of
  every recorded run. Wired correctly it answers `left` on 80/80 real frames,
  and no prompt beats 0.55 against a 0.50 baseline. Every search result in this
  repo is geometric search with the VLM inert.
- **The `unexpected` axis is at ceiling** — D and E both score 1.000 on the one
  episode and disagree on 0 of 176 frames. It cannot discriminate in either
  direction.
- **`clutter_similar` has 136 frames and no manifest entry**, so no
  capture-time ground truth and no way to score it.
- **The live corpus is 94% target-present** — 66 of 1,099 scored frames have
  the target genuinely absent. False belief, the error that matters for
  safety, is barely measurable. A system that always answered "present" would
  score 94%.
- **n = 1 per live scenario, n = 3 per nano condition.** Supports "X fails",
  not "X beats Y by a little".
- One scene, one operator, one camera, one target object throughout.
- Search trials raised `MAX_PER_DIRECTION` from 2 to 6 so success was possible
  at all — **not shipped behaviour**.

Eleven measurement bugs found and fixed are documented in `PROGRESS.md` rather
than quietly corrected. Each produced a confident, plausible, wrong answer
first; most surfaced only because two measurements disagreed.

---

## Captured frames are not in this repo

`live_disruption_frames/`, `nano_frames/` and the generated sets are
gitignored. They are photographs of a private room — desk, an open laptop with
readable documents, a closet, and in some frames the operator. The derived
`*.json` results are tracked and are the actual research artifact.

If frames are ever needed for reproduction, use a separate private repo or Git
LFS — as a deliberate decision, not by accident.
