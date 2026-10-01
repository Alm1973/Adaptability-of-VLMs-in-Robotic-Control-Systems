# Recovering Object Verification on a 2-DOF Camera Arm: What the VLM Actually Contributes

*Consolidated findings report. Grounded in `study_results.json` (synthetic,
SURROUND_SIMILAR=0.67), `live_bench.json` (real frames), `probe_real.json`
(live occluder-variety session), `clip_occluder_real2.json`
(occluder/replacement classifier), `nano_all.json` (hardware search sweep),
`runs/2026-09-01_233533/` (controlled-rig run, `ablation_scores.json`),
and the sweep/latency JSONs referenced inline. Tables were cross-checked
against those files on 2026-08-13 and refreshed 2026-08-28 against five
cycles of results that had landed since (live occluder variety, the
clip_classifier fix, the hardware search sweep, the AMBIGUOUS throttle fix,
and the `far`-asymmetry resolution), then again same day against
Assignments 2–3(b) (`assignment2.json`, `assignment3a.json`,
`assignment3b_offline.json`, `search_prompt_sweep.py`): F_probe's live win is
now known to be a string-match artifact, not architecture; the VLM has never
actually steered a search on hardware; and all remaining live captures move
to a controlled rig (see `LIVE_SESSION_PLAN.md`). **Refreshed again
2026-09-02** against the controlled rig's first full run (48 trials, 8
scenarios x 3 reps x 2 clutter levels, `runs/2026-09-01_233533/`) and its
ablation: the identity finding (§4, previously synthetic-only) is now
confirmed on an independent live corpus with a valid identity episode for
the first time; a new event-gating limitation was found (§5c); and a
scoring bug in the *other* (non-authoritative) scorer was found and fixed.
The authoritative running state remains the living block at the top of
`PROGRESS.md`.*

---

## 1. Research question

> Can a locally hosted VLM, embedded in a reasoning pipeline that uses OpenCV
> for spatial reasoning, **recover** from disruptions to object verification
> caused by **occlusion**, **environment change**, and **unexpected objects**?

Sharpened so a single lucky episode cannot satisfy it: *under which disruption
types does the pipeline restore correct belief, how reliably, and **which
component is responsible?*** The last clause is the load-bearing one — it is
what forces the ablations. Without them one cannot tell whether the VLM, the
OpenCV geometry, or the belief state machine did the work.

**The short answer.** The pipeline recovers from all three disruption families,
but the locally hosted VLM is *not* the general cause. Two components fix
**orthogonal** failures and neither substitutes for the other:

- **OpenCV geometry + a belief state machine** fix **occlusion** (and hold
  through environment change).
- **The VLM** fixes **identity** — same-class / attribute impostors — and
  nothing else measurable. Confirmed synthetically first (§4), and as of
  2026-09-02 confirmed on an independent live corpus with a real identity
  episode (§5c) — no longer a synthetic-only claim.

Search guidance, the third role originally imagined for the VLM, was tested and
**loses to random.**

---

## 2. System architecture

Fully local. Three tiers, with a hard division of labour that was forced by
experiment (see §7, the mosaic-REACQUIRE dead end): **OpenCV owns geometry, the
VLM owns semantics.**

| Tier | Component | Role |
|---|---|---|
| Perception | Open-vocabulary detector (YOLO-World: CNN backbone + FPN) | boxes / classes / confidence for any user-named target |
| Semantics | Qwen2.5-VL-3B (4-bit, transformers runtime, `max_pixels=200704`) | crop verification: "is this the target?"; open probe: "what is in this region?" |
| Geometry | OpenCV | scene-change localisation, IoU, occlusion vs surround, ego-motion / periphery, motion |
| State | Belief state machine | CONFIRMED / OCCLUDED / DISPLACED / MISSING / AMBIGUOUS + recovery policy |

Per user decision 8 the detector is a real perception layer (off-the-shelf,
pretrained, not trained by us), so the system is **YOLO + VLM + OpenCV**, not
VLM + OpenCV. The writeup states this explicitly because it changes what "the
VLM's contribution" means: the detector already solves most per-frame presence.

**Why identity verification is architecturally required, not optional.** The
open-vocab detector matches the *noun and ignores the qualifier*: `"blue cup"`
fires at 0.712 on the *same box* as `"red cup"` (IoU 0.98) in a scene with no
blue cup. A colour- or attribute-named target therefore cannot be confirmed by
the detector alone — something must check identity on the crop. That is a
conclusion from `detector_decision.py`, independent of the ablation.

---

## 3. Method

### 3.1 Corpus
- **Synthetic (primary, per decision 3):** `disruption_bench.py` →
  **36 episodes / 816 frames**, 12 disruption types × 3 randomised repeats,
  disruption length drawn from **{4, 8, 14}** frames. Ground truth is exact *by
  construction* — the bench synthesises the disruption, so it knows the onset
  frame and whether the object is still present. Human-eyeballed before scoring
  (`_contact_sheet.jpg`).
- **Live (validation):** `live_disruption.py` captures scripted real scenarios
  (hand occlusion, lighting, camera pan) with per-frame `present`/`edge` GT
  recorded at capture time. `live_bench.py` scores them with the *same*
  `score_episode` as the synthetic study — one scorer, two corpora, directly
  comparable.

### 3.2 Conditions (ablations)
| | Detector | VLM | State machine |
|---|---|---|---|
| **A** VLM only | — | per-frame | — |
| **B** detector only | ✓ | — | — |
| **C** stateless | ✓ | crops | — |
| **D** full pipeline | ✓ | crops | ✓ (decay clock) |
| **E** = D − VLM | ✓ | — | ✓ |
| **F** = D + region probe | ✓ | crops + open probe | ✓ (probe replaces clock) |

**E is the addition that makes the VLM separable.** C-vs-D cannot isolate the
VLM because C already contains it — that contrast measures the *state
machine*. Only **D-vs-E**, which differ solely in the VLM, isolates the VLM
itself.

### 3.3 Metrics
Aggregates hid every important effect and three separate faked-result bugs, so
everything is **broken out by disruption kind**:
- **durAcc** — belief correctness *while the disruption is active* (not after it
  ends; scoring after the disruption saturated every condition at 7/7 and made
  an earlier study blind).
- **lost-while-present** — dropping an object that is merely hidden.
- **false-belief** — asserting presence when the target is absent
  (safety-critical).
- spurious re-acquisition, stable-phase accuracy.

---

## 4. Results — synthetic corpus (SURROUND_SIMILAR = 0.67)

`durAcc / lost / false`, per kind:

| condition | environment | identity | occlusion | unexpected |
|---|---|---|---|---|
| A VLM only | 0.972 / 7 / 0 | 0.0 / 26 / 0 | 0.333 / 52 / 0 | 1.0 / 0 / 0 |
| B detector | 0.992 / 2 / 0 | 0.0 / 26 / 24 | 0.333 / 52 / 0 | 1.0 / 0 / 0 |
| C stateless | 0.976 / 6 / 0 | 0.0 / 26 / 0 | 0.333 / 52 / 0 | 1.0 / 0 / 0 |
| **D full** | 0.992 / 2 / 0 | **0.952 / 2 / 0** | **0.968 / 4 / 0** | 0.691 / 0 / 24 |
| **E = D − VLM** | 0.992 / 2 / 0 | **0.952 / 2 / 24** | 0.968 / 4 / 0 | 0.691 / 0 / 24 |
| F = D + probe | 0.992 / 2 / 0 | 0.595 / 14 / 0 | 0.786 / 24 / 0 | 0.869 / 0 / 8 |

**Finding 1 — the state machine fixes occlusion.** Every stateless condition
(A/B/C) bleeds 52 lost frames at durAcc 0.333. Adding the belief state machine
(D/E) gives 4 lost at 0.968. The VLM is irrelevant here: D and E are identical.

**Finding 2 — the VLM fixes identity, and *only* identity.** D and E are
byte-identical in every column **except identity**, where the VLM takes
false-belief from **24 → 0** and spurious re-acquisition from **3/3 → 0/3**.
Everywhere else the VLM contributes nothing measurable. This is the headline: a
narrow, real, decisive-where-it-binds contribution — invisible to earlier runs
because no episode then required verification.

**Finding 3 — the state machine has a measured cost.** On `unexpected`
(novel object at the target's spot) both D and E drop 1.0 → 0.691 and gain 24
false-belief frames. A carton placed where the cup was is geometrically
identical to the cup being hidden *behind* a carton, so belief is held through
the decay window. Holding through an occlusion and holding too long after a swap
are the *same mechanism*. The VLM cannot rescue this: verification only fires on
a detection, and a different-class replacement produces the *target's* label
only when geometry already matched — so the semantic check is never the thing
that separates covered from replaced at a single instant. Only **time** can
(a real occlusion ends; a removal does not), which is what motivates the probe.

---

## 5. Results — real corpus, and why it inverts the synthetic ordering

`live_bench.py`, 446 scored frames (219 edge-excluded as genuinely undefined).
The first live pass had only 2 occlusion episodes, both a **hand**; a later
7-scenario session (`probe_real.py`) added book/paper/box/jacket occluders and
changes the picture substantially — see the correction below.

**By kind, occlusion (7 episodes, `probe_real.json`):**

| condition | durAcc | lost | false |
|---|---|---|---|
| A/B/C stateless (hand-only, 2-episode pass) | 0.0 | 80 | 0 |
| D full | 0.299 | 160 | 5 |
| E = D − VLM | 0.293 | 162 | 23 |
| **F = D + probe** | **0.504** | **98** | **5** |

**⚠️ CORRECTION to an earlier draft of this section.** A prior pass reported
F_probe as "1.0 durAcc, 0 lost" on real occlusion. That number was measured on
only 2 episodes and **both were a hand** — the one occluder type the probe's
keyword list actually names. Per-scenario, the real range is much wider:

    scenario           durAcc   lost   what the VLM called it
    occlusion_full      1.000      0   "hands" x5 — correct
    occlusion_remove     1.000      0   (recovers correctly when actually gone)
    occlude_book         0.562     14   "desk lamp"/"purple book"/"notebook"
    occlude_jacket       0.143     24   "socks" x5
    occlude_box          0.129     27   "notebook" x5
    occlude_paper        0.000     25   "document" x5

**F still beats D on every real occlusion scenario measured** (0.504 vs 0.299
by kind), so the synthetic-vs-real inversion below still stands — but F is a
partial fix limited by a hand-written keyword list, not a solved problem. A
jacket read as "socks" and paper read as "document" hold no better than the
plain decay clock, because the probe's classification depends on the VLM's
word matching a list it was never shown these words for. (The `clip_classifier`
fix in §8 addresses exactly this brittleness, but was validated on the *words*
the VLM already produced here, not by re-running these live scenarios end to
end with it wired in — that re-run has not happened.)

**The ordering (D vs F) still reverses relative to synthetic**, and the
mechanism is visible in the state counts on the 61-frame hand-covered episode:

    A/B/C   CONFIRMED 96, MISSING 61            belief dropped instantly
    D/E     CONFIRMED 96, OCCLUDED 12, MISSING 49   held for exactly DECAY_FRAMES
    F       CONFIRMED 96, OCCLUDED 61            held the whole way

Synthetic disruptions are 4/8/14 frames; the decay clock is 12, so **it never
expired on the synthetic corpus** — the longest synthetic disruption is shorter
than the parameter it was meant to test. Real occlusions run 40–61 frames and
always outlast it. The **region probe** ("what IS in this region?" — a hand ⇒
still covered, a carton ⇒ replaced) replaces the clock and holds belief for the
whole real occlusion when it correctly names the occluder, and still drops to
MISSING 60/60 when the cup is actually removed (`probe_test.py`: 28/30;
`replay_live.py`: 12/61 → 61/61 on the hand episode specifically).

**⚠️ CORRECTION (2026-08-28, Assignment 2, `assignment2.py`): F's live win over D
is a string-match artifact, not architecture.** Four tests settle it:

| test | occlusion_full | occlusion_remove | aggregate | lost |
|---|---|---|---|---|
| F_probe, current vocabulary | 1.000 | 1.000 | 0.504 | 98 |
| **F_probe, `hand`/`finger`/`palm`/`wrist`/`arm` removed** | **0.000** | **0.000** | **0.141** | 192 |
| D_full (decay=12) | 0.050 | 0.050 | 0.299 | 160 |

The VLM still answers "hands" on every frame; deleting five words from the
lookup list collapses the score from 0.504 to 0.141 — *below* D, not merely
back to it, because a probe that cannot name the occluder concludes
"replaced" and drops belief immediately, where the timer at least waits 12
frames. **A probe that fails is worse than no probe.**

The architectural framing dissolves further under a timer sweep with the
probe removed entirely:

| `decay_frames` | aggregate | occlusion_full | false belief |
|---|---|---|---|
| 12 (shipped) | 0.299 | 0.050 | 5 |
| 20 | 0.445 | 0.250 | 5 |
| 30 | 0.554 | 0.500 | 5 |
| 45 | 0.661 | 0.875 | 5 |
| **60** | **0.697** | **1.000** | 5 |
| infinite | 0.697 | 1.000 | 5 |

Changing one integer from 12 to 60 beats F's entire probe mechanism (0.697 vs
0.504) and matches it exactly on the hand episodes. Giving the probe a
perfect vocabulary (the exact nouns the VLM produced) or an oracle answer
both land on **0.697 / 58 lost / 5 false — identical to the plain timer.**
Three different routes converge on the same ceiling; the timer is the
cheapest one to reach it. **`occlude_paper` scores 0.000 in all eleven
configurations tested, including the oracle probe** — that residual is the
detector never re-acquiring after the paper is lifted, a detection failure
no belief policy touches.

**The real story is not "sim doesn't transfer to reality," it is a
mis-calibrated parameter:** the synthetic corpus's longest disruption is 14
frames, shorter than `DECAY_FRAMES=12` was ever asked to survive, so 12 was
never really tested before shipping. Real occlusions run 40–61 frames and
always outlast it. Raising the timer is the fix `run_study.py`'s ablation
never surfaced, because the synthetic corpus cannot generate a disruption
long enough to need it.

**The caveat that stops "raise the timer to 60" from being a safe
recommendation: false belief is 5 at every timer value tested, including
infinite.** Never conceding the target is gone should generate false belief
constantly — it doesn't, because the live corpus is **94% target-present
(66 of 1,099 scored frames absent)**. The exact tradeoff the timer and the
probe both exist to manage — hold when hidden, drop when actually gone — is
the one thing this corpus cannot measure. So the honest claim is **"the
probe is not justified by this corpus,"** not "the probe is useless"; on an
absence-heavy corpus, infinite patience should fail badly and the probe may
earn its cost after all. This is why an absence-heavy capture is now the
project's top open item (§9, `LIVE_SESSION_PLAN.md`).

**Synthetic occluders are not a valid proxy for a hand.** Five independent
demonstrations accumulated across cycles:
1. a real hand trips the ego-motion branch (a large near-lens object moves the
   frame like a pan); a synthetic rectangle never does — fixed by
   `_periphery_diff` (does the *far background* move? still≈15, hand≈22, pan≈74,
   threshold 48);
2. synthetic `camera_pose` keeps the target visible, so `_diagnose` is never
   reached and the DISPLACED path is untested synthetically in both directions;
3. the probe reads a flat rectangle as a *replacement* and a hand as an
   *occluder* — opposite verdicts on the same nominal disruption type;
4. `SURROUND_SIMILAR` had to be recalibrated on real frames (0.40 → **0.67**);
   the synthetic sweep's tradeoff never fired because flat patches score ~0.1
   while a textured, shadow-casting hand scores 3–5× higher;
5. the decay clock is well-tuned only because the synthetic disruptions are
   shorter than it.

**Report occlusion results from the synthetic corpus as a rectangle-shaped
disruption, not as occlusion.**

---

## 5b. Assignment 3(a): a frame-level audit of where the VLM's contribution lives

The D-vs-E table in §4 is an aggregate. `assignment3a.py` re-ran it **per
axis**, counting VLM calls and diffing the two belief trajectories
frame-by-frame, so a tie caused by two pipelines *agreeing* cannot be
mistaken for a tie caused by them *cancelling out*.

| axis | n | D durAcc | E durAcc | D false | E false | D lost | E lost | VLM calls | frames disagreeing |
|---|---|---|---|---|---|---|---|---|---|
| environment | 2 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 2 | **0 / 339** |
| unexpected | 1 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 1 | **0 / 176** |
| occlusion | 7 | 0.299 | 0.293 | **5** | **23** | 160 | 162 | 22 | 121 / 1098 |

**On environment and unexpected, D and E never diverge — 0 of 515 frames.**
The VLM was consulted 3 times total and changed nothing, measured at the
frame level rather than inferred from a tied average. OpenCV alone suffices
on these axes; this is the cleanest statement of that claim in the project.

**On occlusion the accuracy delta (+0.006) is noise. The signal is entirely
in the false-belief column (5 vs 23),** and it comes from one episode:

| condition | durAcc | lost | false belief |
|---|---|---|---|
| E_no_vlm | **0.923** | 2 | **23** |
| D_full | 0.615 | 10 | **5** |

**Removing the VLM raises accuracy and multiplies false belief by 4.6.**
Ranking these two conditions on accuracy alone selects the more dangerous
pipeline and produces a number that *justifies* the choice — the strongest
argument in the project for reporting false belief beside accuracy and never
folding it into a single score. On the two hand episodes specifically, D and
E diverge on zero frames, consistent with §5's finding that the VLM
contributes nothing there either.

**Limit:** the `unexpected` axis is at ceiling (1.000/1.000 on a single
episode) and cannot discriminate in either direction — a corpus limitation,
not a VLM finding, and the reason a harder, more cluttered scene is on the
plan (`LIVE_SESSION_PLAN.md`) before any claim about "unexpected objects in
clutter" is attempted.

---

## 5c. The controlled rig: identity confirmed live, and a new blind spot found

`final_experiment.py`, run `2026-09-01_233533` (operator-attended, camera-only —
the arm did not move except homing). 8 scenarios x 3 repetitions x 2 clutter
levels (none/high) = 48 trials, 8428 frames, 5699 scored, **691 absent
(12.1%)**, against 6.0% in the desk pilot. Zero missing frames. `score_run.py`
replayed the identical frames with the VLM switched off (D_full vs E_no_vlm)
plus a decay-timer sweep.

| axis | n | D acc | E acc | Δacc | D false | E false | VLM calls |
|---|---|---|---|---|---|---|---|
| control | 6 | 1.000 | 1.000 | 0.000 | 0 | 0 | 6 |
| environment | 12 | 1.000 | 1.000 | 0.000 | 0 | 0 | 12 |
| occlusion | 18 | 0.516 | 0.520 | −0.004 | **60** | **118** | 63 |
| identity | 6 | 0.183 | 0.301 | **−0.118** | **36** | **76** | 39 |
| unexpected | 6 | 0.392 | 0.358 | +0.034 | **100** | **103** | 12 |

Totals: D_full 196 false-belief frames, E_no_vlm 297, over 132 calls across
48 episodes (2.8 per episode). The raw difference is 101 frames, but see
§5d before quoting it: that figure is an average over trials most of which
show no difference at all, and it is not the defensible form of the claim.

**§4's H1 (the VLM's contribution is confined to refusing false presence, not
to holding belief through occlusion) holds on independent hardware data,
three ways:**

1. **Control and environment are identical between D and E** — 18 episodes,
   18 VLM calls, zero difference in any column. This replicates the pilot's
   0-of-515-frames result (§5b) on a corpus captured under different
   conditions (fixed lighting, marked target position, clutter as a factor
   rather than an accident).
2. **Identity: false belief roughly halves when the VLM is removed (76 → 36),
   while accuracy *falls* (0.301 → 0.183).** Removing the VLM makes the
   system look better on the metric that ignores direction and behave worse
   on the one that matters. This is the same signature the synthetic corpus
   showed in §4 Finding 2 (false-belief 24 → 0 with the VLM present) —
   except this time it is measured on a live identity episode. **The pilot
   corpus had zero valid identity episodes after three failed capture
   attempts (§9); this run supplies the first one**, so the identity claim in
   the short answer (§1) is no longer resting on synthetic data alone.
3. **The occlusion row is not an occlusion result.** Split by scenario, the
   whole of it is `removal`:

   | scenario | D false | E false |
   |---|---|---|
   | occlude_hand | 0 | 0 |
   | occlude_object | 0 | 0 |
   | removal | 60 | 118 |

   `removal` sits in the occlusion axis because `live_bench.KIND` files it
   there, and that mapping is wrong for this purpose: the cup is taken away,
   so the episode ends absent. Actual occlusion contributes nothing to either
   side. An earlier draft of this section reported the axis aggregate as an
   occlusion finding. It is not one, and the claim is withdrawn.

**New finding: the VLM cannot help where the pipeline never asks it.**
`substitution` (a pink pair of pliers standing where the red cup was) is the
one axis where the VLM does almost nothing: **103 → 100, three frames**, the
worst false-belief rate of any scenario in *either* condition. The mechanism
is visible in calls-per-episode:

| axis | VLM calls / episode |
|---|---|
| identity | 6.5 |
| occlusion | 3.5 |
| unexpected (substitution) | 2.0 |
| environment | 1.0 |

The VLM is *event-gated*: the pipeline calls it when the detector's evidence
looks wrong. A pink object of about the right size in exactly the right
place does not look wrong to a detector prompted with `"red cup"` — YOLO-World
keeps returning a confident box on it — so nothing escalates and the VLM is
never asked. **The gating that keeps the VLM affordable (2.8 calls/episode,
~3.4% of runtime) is the same mechanism that makes it blind to any failure
the detector does not flag.** This is a structural property of the
tier split (§2), not a weakness of the model, and it did not show up before
because the pilot's `impostor`/`distractor` episodes never isolated a case
where the detector itself was fooled.

**The decay-timer question from §5's caveat, answered with data that can
actually punish patience.** §5 could not test whether `DECAY_FRAMES=60` is
safe, because the pilot corpus was 94% target-present and false belief sat at
5 for every timer value tested, including infinite. At 12.1% absent:

| timer | occlusion acc | occlusion false | unexpected acc | unexpected false | total false |
|---|---|---|---|---|---|
| 12 (shipped) | 0.516 | 60 | 0.392 | 100 | **196** |
| 30 | 0.667 | 61 | 0.675 | 128 | **229** |
| 60 | 0.831 | 61 | 0.958 | 134 | **235** |

Raising the timer is **not free**: +39 false-belief frames (+20%) going from
12 to 60. But the cost is not evenly distributed:

- **On occlusion it is nearly free** — accuracy 0.516 → 0.831 for **+1**
  false-belief frame. The target really is behind something, so patience is
  simply correct.
- **On unexpected it is expensive** — +34 false-belief frames for the
  accuracy gain. The target really is gone, so patience is simply wrong.

So the right fix is not a single global constant — it is a **longer timer
conditioned on diagnosed state** (patient under OCCLUDED, short under
evidence of substitution), which the belief state machine already has the
information to compute. This is a concrete, data-backed design change that
the pilot's 94%-present corpus could not have produced.

**A 13th measurement bug, found by this run.** `final_experiment.score_trial`
computed accuracy over `disrupt` **plus** `recover`, while `run_study.score_episode`
— the scorer behind every other number in this writeup — uses the disrupt
phase only; the docstring claimed they matched. The recover phase is the easy
part (disruption over, cup in plain view), so including it inflates accuracy:
`identity_swap` read 0.513 live against **0.183** from the shared scorer.
`false_belief` and `lost_while_present` were computed identically in both
scorers, so **the primary (safety) metric was never affected — only
accuracy, and only in the lenient direction.** Fixed, with a regression test.
**All numbers in this section are from the shared, disrupt-phase-only
scorer**, not the live run's on-the-fly verdicts.

**Caveats specific to this run:**
- 12.1% absent still trips the project's own 15%-absence warning threshold —
  better than the pilot's 6.0% and enough to expose the timer tradeoff above,
  but the ≥40%-absent corpus in `LIVE_SESSION_PLAN.md` item 1 is still worth
  doing.
- n = 6 per scenario, 3 per scenario x clutter cell — enough to support the
  *direction* of these effects, not a fine ranking between them.
- Control and environment sit at 1.000 for both conditions in this rig too —
  still a ceiling, still cannot discriminate.
- The occluder was a single generic object (a cardboard box) by operator
  decision, so the per-occluder ordering from §9 remains unrankable; this run
  does not settle it.
- Live verdicts and offline (replayed) scores differ slightly on borderline
  frames (`identity_swap` false belief: 36 replayed vs 33 live) — lossy JPEG
  compression shifts detector confidence near the threshold on the ~9% of
  frames that are close calls. Offline is authoritative: it is the only
  condition where every arm sees byte-identical input.

---

## 5d. How much of §5c survives a per-trial check

The axis table in §5c averages over 48 trials. Averages hide dispersion, and
here the dispersion is the finding. Per-trial `E_no_vlm − D_full` false-belief
counts:

| scenario | per-trial difference | trials where the VLM changed anything |
|---|---|---|
| occlude_hand | 0, 0, 0, 0, 0, 0 | 0 of 6 |
| occlude_object | 0, 0, 0, 0, 0, 0 | 0 of 6 |
| removal | 0, 37, 0, 21, 0, 0 | 2 of 6 |
| substitution | 0, 3, 0, 0, 0, 0 | 1 of 6 |
| identity_swap | 0, 26, 0, 3, 0, 11 | 3 of 6 |

**42 of 48 trials show no difference between the two conditions.** Two trials
(37 and 26 frames) account for 62% of the 101-frame total. So "the VLM
prevents 34% of false belief" is an average over a distribution that is mostly
zeros with two large values, and the percentage should not be quoted as an
effect size.

### What is defensible

The direction, and it is defensible on its own terms. In all six trials where
the two conditions differ, `E_no_vlm` is worse. The VLM never once increased
false belief. Under the null hypothesis that it has no effect, the sign of
each non-zero difference is a coin flip, so six out of six in the predicted
direction gives p = 0.5^6 = 0.016.

That supports **"the VLM reduces false belief, and never increases it."** It
does not support any statement about magnitude.

### The confound

Detection rate varies enormously within a single scenario, and the variation
is not random with respect to the outcome:

| scenario | detection rate per trial (%) | spread |
|---|---|---|
| control, lighting, camera_pose | 100 in every trial | 0 |
| occlude_hand | 66, 66, 66, 69, 100, 100 | 34 |
| occlude_object | 65, 66, 94, 95, 96, 100 | 35 |
| identity_swap | 35, 37, 38, 41, 46, 68 | 33 |
| removal | 32, 33, 33, 87, 89, 100 | 68 |
| substitution | 32, 32, 33, 42, 97, 100 | 68 |

The two absent-ending scenarios split into two groups. In the trials near 32%
the detector stops firing once the cup leaves, and both conditions record
almost no false belief because there is nothing to be wrong about. In the
trials near 90 to 100% the detector keeps returning a confident box, and that
is where the false-belief counts of 36 to 40 appear.

**What predicts false belief in this run is whether the detector kept firing
after the target left, not which condition was scored.** That variable was not
controlled. It depends on how completely the cup left the frame and what was
left behind, which is staging, and it varies by 68 points inside a single
scenario name.

Two consequences. Within a trial the comparison is still clean, because
D_full and E_no_vlm read the same frames, so staging cannot explain a
difference between them. Across trials the aggregate is dominated by whichever
trials happened to land in the high-detection group, which is why the mean is
unstable at n = 6.

### Status of the §5c claims after this check

| claim | status |
|---|---|
| Control and environment identical between D and E | holds, 18 of 18 trials |
| Occlusion produces no false belief in either condition | holds, 12 of 12 trials |
| The VLM reduces false belief and never increases it | holds, sign test p = 0.016 |
| Identity: false belief halves while accuracy falls | direction holds, magnitude rests on 3 trials |
| The VLM is barely called on substitution | holds, call counts are mechanical |
| The VLM prevents 34% of false belief | **withdrawn as an effect size** |
| Occlusion false belief halves | **withdrawn, see §5c item 3** |

### What would settle it

One session of absent-ending scenarios only, `removal`, `substitution` and
`identity_swap` at 8 repetitions each, 24 trials and about 35 minutes. The
present-ending scenarios cannot produce false belief and consumed three
quarters of this run's time producing structural zeros.

Detection state after the target leaves should be recorded as a variable
rather than left to vary. `live_view.py` already shows the detector's verdict
live, so the operator can log which group a trial falls into before recording
the recover phase instead of discovering it afterwards.


## 6. Search guidance: the VLM loses to random

Static episodes cannot move a camera, so a **virtual pannable camera** (window
cropped from a larger world image, panned by the pipeline's own chosen
direction — closed loop, exact GT, no hardware) tested decision 1's core claim
that "the VLM decides where to look" (`run_search_study.py`):

| strategy | found | rate | VLM calls |
|---|---|---|---|
| vlm | 3/7 | 0.429 | 63 |
| sweep (fixed pattern) | 3/7 | 0.429 | 0 |
| vlm_eps (ε-greedy) | 25/35 | 0.714 | 161 |
| **random** | **27/35** | **0.771** | 0 |

**Mechanism: the VLM commits.** It answered "left" three times running and swept
the base into the 0° stop; committing to a wrong direction burns the
per-direction budget. Epsilon-greedy genuinely fixed the commitment failure
(0.429 → 0.714) — but random matches it for free, so the gain came from the
*randomness added to the VLM, not the VLM*. `SEARCH_POLICY` defaults to
`"random"` (switchable to `"vlm_eps"`). *(Warning kept: the first search run was
invalid — starts were physically unreachable within the per-direction cap, so
all strategies "failed" for reasons unrelated to direction choice. The harness
now refuses unreachable starts. Always check a negative result is not just an
impossible task.)*

**This test is unaffected by §8's `search_direction()` correction.**
`run_search_study.py` builds its verifier with `run_study.make_verifier()`,
the `.generate()`-based verifier that decodes text, not
`make_fast_verifier()`'s yes/no comparator — so the VLM here was genuinely
connected and its "left" commitment is a real answer, not a parse failure.
The two results are complementary, not contradictory: a *connected* VLM loses
to random on direction choice (this section), and separately, on real
hardware the VLM was never even connected to try (§8) — both are true, about
different code paths.

---

## 7. Two user decisions now contradicted by data (flagged, not overridden)

- **Decision 1 (VLM-guided search):** tested and it loses to random (§6). The
  *other half* of decision 1 is **confirmed** — the fixed sweep pattern the user
  rejected ties with the VLM and also loses to random.
- **Decision 6 (derive the OCCLUDED→MISSING timeout from data):** answered in
  the **negative**, which is itself the useful result. `decay_sweep.py`
  stratified by disruption length: best decay per stratum is 4/8/12 for
  dlen 4/8/14 — the optimum *tracks the disruption length*, which the robot
  cannot know in advance. Run 4's "optimum = 8" was the bench reciting its own
  `DISRUPT_FRAMES = 8` constant. **A fixed timeout is the wrong *shape* of
  answer**; the region probe is the proposed replacement.

Decisions 2 (arbitrary targets — verified live: red cup 0.96, keyboard 0.88,
mouse 0.60, phone 0.39), 4 (sign/mirror fixed on the firmware side), 5
(re-verify identity before declaring recovery), 7 (overwrite in place, preserve
the generation-counter fix), and 8 (real perception layer) are honoured.

---

## 7b. The occluder-word classifier, and the AMBIGUOUS throttle

**The keyword list (`OCCLUDER_WORDS`) fails on 18/25 real occluder instances
it wasn't told about** — a jacket named "socks", a box named "notebook", paper
named "document" (§5). A CLIP-text classifier scored against the VLM's own
word fixes the *listed-vocabulary* brittleness: `kw` 7/25 vs `clip_text`
**23/25** on real (non-inpainted) occluder words, with dangerous errors
(false "still there") unchanged at 2/38 — those 2 are hand-placing-the-impostor
transition frames, which are temporal, not perceptual, and no single-instant
classifier can fix them. `clip_classifier.py` (`make_clip_classifier`) is
wired into `RecoveryPipeline.probe_classifier` and `find_live.py`'s
constructor call, confirmed end-to-end (not just scored on logged words) by
`probe_integration_check.py`: **28/30** scenario-majority-correct, weakest on
`occlude_book` (3/5 — "desk lamp" still wins some votes).

**A second, unrelated cost bug was found by a hardware sweep, not by
reasoning about the code.** `nano.py`'s 16-condition search-degradation sweep
found `box`/`cloth`/`shadow`/`glare` trials burning **85–104 VLM calls in a
single ~15–30 s trial** — because outside CONFIRMED, *any* detector hit called
`_verify()` with no throttle, unlike OCCLUDED's `probe_every=8`. Fixed with a
`verify_every=8` / `_reject_streak` counter, same pattern as the existing
throttle. Re-running D_full/F_probe on the full 36-episode synthetic corpus
confirmed durAcc/lost/false are **byte-identical** before and after — pure
cost fix — while total VLM calls fell (D: 64→43, F: 78→57 over 36 episodes),
all of the saving landing on `identity_same_class` episodes where the detector
correctly keeps re-finding an impostor the VLM correctly keeps rejecting.

## 8. Supporting engineering results

- **Model choice (Phase 0).** The `@@@@` degenerate lock was a *llama.cpp CUDA
  vision-kernel bug, not a model bug*; switching the runtime to
  transformers/PyTorch removed it (0/25 degenerate). The 15× speedup came from
  **capping vision tokens** (`max_pixels=200704`), not a new model: uncapped
  5.62 s → 0.58 s at unchanged accuracy 0.812. No new model was needed.
- **Vision budget = 200704, validated not inherited.** `pixel_budget.py`: peak
  VRAM is flat (~2511 MiB) at every budget, so "free VRAM to buy a bigger
  budget" was wrong at its premise; latency rises 4.4× to 602112 for zero
  accuracy gain. Caveat: the crop set is saturated (acc 1.000 everywhere), so
  this shows 200704 is *not too small*, not that 100352 is safe —
  `budget_discriminate.py` then showed both pass 24/24 on a hard set.
- **VLM latency — quote *within-run ratios only*.** Logits (single forward pass)
  vs generate_8 is 1.27× with byte-identical answers (safe for the ablations);
  cumulative ~1.5×, and search now costs *zero* VLM calls. Absolute latency is
  **not stable across runs** (the same config measured 1.671 s and 0.644 s with
  no code change), so no absolute number is quoted.
- **Motion smoothness (hardware).** `controller.py` rewritten (continuous
  deadzone, adaptive smoothing, slew limit): 0 commands vs 14 on a static
  jittery target, better final centring. Firmware rewritten and flashed
  (eased ramp at 83 Hz), so the servo interpolates between sparse ~10 Hz
  commands — something no host-side fix can do.
- **Hardware search-degradation sweep (`nano.py`, 16 conditions x 3 repeats,
  triage-level n).** Brightness robustness beat the file's own prediction: the
  cup was found down to 0.06x software brightness in every valid trial — what
  degrades is *centring* (20.9 s / 20 moves to converge vs sub-second at full
  brightness), not detection, so found/centred must be reported separately.
  Full opacity (`box`, `cloth`) fails completely and *safely* (0/3 found, never
  a false CONFIRMED) but was expensive before the AMBIGUOUS throttle fix above.
  The same-class-impostor safety check (decision 5) held on the *search* path
  for the first time, not just static verification: all 3 `impostor` trials
  correctly ended non-found. `far`'s two-trial-vs-one-trial directional split
  (0/2 at `-50` deg, 1/1 at `+50` deg) turned out to be **room geometry, not a
  search-capability limit** — resolved 2026-08-28 by inspecting the saved
  per-trial frames (`nano_frames/far_*.jpg`), which showed the `-50` deg
  direction pointing at an unrelated monitor that could never contain the cup,
  while `+50` deg landed on it directly with zero search moves. No new
  hardware trials were needed to settle this; the frames already on disk
  answered it once someone looked.
  **⚠️ CORRECTION (2026-08-28, Assignment 3(b)): every result above is
  geometric search, not VLM-guided search.** `nano.py` builds its verifier
  from `find_live.make_verifier()`, which is `make_fast_verifier()` — a yes/no
  logit comparator, same as the live pipeline. `RecoveryPipeline.search_direction()`
  asks it a four-way "left/right/up/down" question; the comparator can only
  answer `'no'`, no direction word parses, and the function returns `None` on
  **24/24** logged frames — confirmed the only path ever taken, in every nano
  trial this project has recorded. The VLM has never chosen a search
  direction on hardware; every number above, including the `far` resolution
  and the `impostor` safety check, was produced by the geometric fallback
  with the VLM inert. The underlying findings stand as *geometric*-search
  results; only the "VLM-guided" label was wrong. `search_direction()` is
  left deliberately disconnected rather than rewired, because wiring it to
  `.generate` does not fix it either: it answers **`left` on 80/80** real
  frames (40 clean, 40 cluttered) despite giving accurate, varied captions of
  the same frames when asked to describe them — it sees, it just answers this
  question the same way regardless. A follow-up prompt sweep
  (`search_prompt_sweep.py`, 6 wordings x 40 balanced left/right crops,
  chance = 0.50) ruled out wording as the cause: nothing beats **0.55**, and
  the phrasing that states which half of the scene is shown scores **0.075**
  — below chance — by echoing the direction word handed to it in the prompt
  rather than reading the image. **The capability is absent, not
  disconnected**, so reconnecting it is left undone until a direction
  mechanism clears 0.50 offline (`LIVE_SESSION_PLAN.md` item 5, BLOCKED).
  This is the same shape of failure as the `OCCLUDER_WORDS` string match
  above: twice now a capability attributed to the VLM turned out to be a
  degenerate mechanism sitting in the right place, surviving because its
  failure mode was silent (`None` looks identical to "search found nothing").
- **Controlled-rig hardware verification (`hw_check.py`, 2026-09-01,
  operator-attended).** Confirmed channel 1 drives the shoulder, not a
  pan/tilt "wrist tilt" as the firmware's own constant names claimed — the
  firmware had flagged its own uncertainty and guessed wrong. Two
  consequences for how results should be read: the shoulder's low range is
  conservatively clipped (~30% of downward travel unreachable, not unsafe,
  just unexplained until now), and **the "tilt" axis translates the camera's
  position rather than rotating its heading**, because the shoulder swings
  the whole arm. `_periphery_diff` and the DISPLACED diagnosis were
  calibrated empirically against this hardware, so their thresholds are
  unaffected, but the *pan/tilt head* mental model behind them was wrong and
  is corrected here. Direction pairing (`arm.update(+10,0)` swings the
  camera right; `arm.update(0,+10)` moves the view down) and camera index
  (`CAMERA_INDEX=1`, the C270, confirmed against index 0) were both verified
  correct. The check itself had a bug — it opened `cv2.VideoCapture` directly
  instead of through the project's `Camera` class and scored the first frame,
  which is solid black on the C270 (documented since 2026-08-08 as the reason
  `Camera._warm_up()` exists) — so it briefly reported the camera *and* the
  VLM broken when both were fine; fixed to require `cam.warm` before
  proceeding. This surfaced `config.YOLO_TARGET` left at `"computer monitor"`
  from an early, unrelated smoke test while every experiment script in the
  project hardcodes `"red cup"` — fixed the same day, code-only, no hardware
  touched for the fix itself.
- **Rejected R&D (kept so it is not re-tried):** motion-as-occluder-signal
  (every in/out ratio ≈ 1.0 — a held hand is not measurably more mobile than a
  placed object at ~10 fps); a constrained two-way probe prompt ("PERSON or
  ITEM", 27/30 vs the open question's 28/30, and its extra errors ran in the
  *dangerous* direction). The mosaic-REACQUIRE approach of asking the VLM for
  precise coordinates was unreliable — this is *why* OpenCV owns geometry.

---

## 9. Threats to validity — what is NOT established

- **n is small and single-scene.** 36 synthetic episodes but one scene, one
  target, one base frame. Live: n=1 per scenario (2 occlusion episodes, both a
  hand; 1 lighting; 1 camera pan). This supports one-directional failure claims
  ("D fails on real occlusion in a way the synthetic corpus hides") but does
  **not** support ranking conditions on small gaps.
- **RESOLVED 2026-09-02, was open for five prior cycles: the desk corpus had
  *no* identity episode** — the one thing the VLM demonstrably does. Live
  `impostor` was misnamed (a different-object swap, i.e. `unexpected`, not a
  same-class impostor) *and* its GT was wrong (the operator set the cup down
  elsewhere in frame, so every "false belief" was the pipeline correctly
  seeing a cup that was really there); excluded via `BAD_GT`. The controlled
  rig's `identity_swap` scenario (§5c) is the first valid one: false belief
  halves and accuracy *falls* when the VLM is removed, the same signature §4
  found synthetically. **The VLM's single distinctive capability has now been
  tested on real frames, n=6 episodes** — still small, still one scene, so
  this is a confirmation of direction, not a settled effect size.
- **Hue-swap impostors probe attribute identity ("red cup"), not shape identity
  (mug vs a different mug).** Do not overclaim.
- **The keyword-list occluder classifier is real-data confirmed brittle, and a
  fix is shipped but not re-validated live end-to-end.** `occlude_book` /
  `occlude_paper` (and `occlude_box` / `occlude_jacket`) DID run
  (`probe_real.py`, 2026-08-15): the keyword path scores 1/5 on real occluder
  naming — only hands. `clip_classifier.py` fixes this on the *words already
  produced* (23/25 vs 7/25) and is wired into `find_live.py`, but the four
  live occluder scenarios above have not been *re-run* through the fixed
  classifier — only replayed offline against logged words. A live re-run is
  the natural next step and has not happened.
- **"GT by construction" holds only if the human followed the protocol.** The
  live impostor error is the project's third labelling bug and the first whose
  aggregate looked completely plausible — verify, don't trust.
- **PARTIALLY ADDRESSED 2026-09-02: the pilot desk corpus was 94%
  target-present (66 of 1,099 scored frames absent), which pinned false
  belief at 5 across every timer value tested including infinite (§5) — the
  one error a longer timeout should generate constantly was unmeasurable.**
  The controlled rig's first run reaches 12.1% absent (691 of 5,699) and this
  was enough to show the timer tradeoff is real and axis-dependent (§5c: +1
  false-belief frame raising the timer on occlusion, +34 on unexpected, for
  the same accuracy gain). **12.1% still trips the project's own 15%-absence
  warning and is short of the ≥40% target**, so a dedicated absence-heavy
  capture (`LIVE_SESSION_PLAN.md` item 1) remains the highest-value open item
  — now to sharpen a tradeoff already shown to be real, not to make an
  unmeasurable one measurable for the first time.
- **Per-occluder ordering (book > jacket > box > paper) is not rankable.**
  Each cell is n=1 episode with 25–32 frames; the gaps between occluders are
  smaller than the variation already seen between repeats of the *same*
  occluder elsewhere in this project. Reported as a description of what
  happened, not a ranking of occluder difficulty. **Still true in the
  controlled rig** — the 2026-09-02 run used a single generic occluder (a
  cardboard box) by operator decision, so the occluder-diversity factorial in
  `LIVE_SESSION_PLAN.md`'s rig spec has not actually been run yet; the rig
  exists and one baseline pass through it is done, but the per-occluder
  question itself is still open.
- **The desk corpus carries uncontrolled variables that a plain scene
  wouldn't.** Two 2026-08-28 findings made this concrete enough to act on: D
  and E agree on 0/515 frames on two whole axes (the corpus cannot
  discriminate there), and the per-occluder ordering above is unrankable at
  n=1. **All remaining live captures move to a controlled rig** — fixed
  camera distance, measured lighting, marked target position, and clutter as
  a deliberate 0/low/high factorial instead of an incidental desk state
  (`LIVE_SESSION_PLAN.md`). The existing desk corpus is kept as pilot data
  that generated these hypotheses, not pooled with rig data, and not
  discarded.

---

## 10. What works on hardware today (verified)

| command | what it does | measured |
|---|---|---|
| `yolo red cup` | find + follow any named object, live window | 9.6 fps, 99% detection, re-homes on exit |
| `findit red cup` | the recovery pipeline live — holds belief through occlusion | 8.9 fps, 4 commands, 1 VLM call / 226 frames |
| `ask what's on the desk` | describe / reason / follow-ups | 2.9 s first, 0.5 s follow-up |

`findit` is `yolo` plus memory: the arm moves only on CONFIRMED, and OCCLUDED
deliberately *holds* position instead of scanning away from an object that never
left. `notify_self_motion()` (verified live 41/41 on a commanded pan) prevents
the pipeline from diagnosing its own corrections as a disruption. The region
probe is live in `find_live.py`; it is off for synthetic scoring because it is
asked to name a pasted rectangle there.

---

## 11. Conclusion

The pipeline **does** recover from occlusion, environment change, and unexpected
objects — but the answer to *which component is responsible* is the point:

1. **The belief state machine + OpenCV geometry** carry recovery from occlusion
   and environment change. **The VLM is irrelevant here** (D ≡ E).
2. **The VLM's sole measurable contribution is identity verification** of
   same-class / attribute impostors (false-belief 24 → 0 synthetically), where
   it is decisive and nothing else substitutes. **As of 2026-09-02 this is
   confirmed on real frames** (§5c): the controlled rig's first valid identity
   episode shows the same signature live — false belief roughly halves when
   the VLM is removed (76 → 36) while accuracy *falls* (0.301 → 0.183) —
   closing a gap that had been open since the first live corpus and survived
   three failed capture attempts.
3. **VLM-guided search loses to random when it is actually connected**
   (§6, `run_search_study.py`, generate-based verifier) — **and on real
   hardware it was never connected at all.** `search_direction()` has
   returned `None` on every logged frame of every nano trial, because the
   live pipeline's verifier is a yes/no logit comparator that cannot decode a
   direction word; every "VLM-guided search" result this project has reported
   on hardware, including the `far` resolution below, was produced by the
   geometric fallback alone. Fixing the wiring does not help: connected to a
   text-decoding verifier it answers `left` on 80/80 real frames, and across
   six prompt formulations and 240 balanced trials nothing beats 0.55 against
   a 0.50 baseline (one formulation scores 0.075 — *below* chance — by
   echoing a direction word out of its own prompt). **The capability is
   absent, not merely disconnected**, and reconnecting it is deliberately left
   undone until something clears 0.50 offline.
4. **A fixed decay timeout does not generalise** in the sense decision 6
   asked (the optimum tracks disruption length, which is unknowable in
   advance) — **but as a plain number it beats the region probe outright on
   the live corpus.** F_probe's apparent win (0.504 vs D's 0.299) was
   measured to be a *string-match artifact*: the VLM says "hands", the
   keyword list contains "hand", and removing five words from that list drops
   F's score to 0.141 — below D, not back to it. Simply raising
   `DECAY_FRAMES` from 12 to 60 reaches 0.697, matched exactly by both a
   perfect occluder vocabulary and an oracle probe answer. Three routes to the
   same ceiling; the cheapest one gets there. This does **not** mean the probe
   is a dead end — false belief is pinned at 5 for *every* timer value tested,
   including infinite, only because the corpus is 94% target-present and can
   barely generate the error the probe exists to prevent. The recommendation
   is therefore "raise the timer, and build a corpus that can tell you if
   that's safe" — not "raise the timer."
5. **The synthetic corpus systematically misrepresents occlusion** (five
   independent demonstrations). Its occlusion numbers describe a rectangle, not
   a hand, and must be reported as such.
6. **A VLM-call cost bug (no throttle outside CONFIRMED) was found on
   hardware** — up to 104 calls in a single trial — and fixed without changing
   any accuracy number (verified by re-running the full synthetic ablation).
7. **A search-hardware finding that looked like a capability limit (`far`'s
   directional asymmetry) turned out to be room geometry**, resolved by
   inspecting frames already on disk rather than running more trials — a
   reminder that not every open question needs new data before it needs
   someone to look at the data that already exists.
8. **The same failure pattern appeared twice, independently.** A capability
   attributed to the VLM (naming an occluder; choosing a search direction)
   turned out both times to be a degenerate mechanism sitting in the place a
   real capability would occupy — a string match on one word, a `None` that
   looks identical to "nothing found." Both survived because their failure
   mode was silent. Worth stating as a pattern in its own right, not filing as
   two unrelated bugs.
9. **Remaining live work moved to a controlled rig**, not the desk. Two
   2026-08-28 findings showed the desk corpus could not support the next
   round of claims: D and E agree on every one of 515 frames across two
   disruption axes (nothing to discriminate), and the per-occluder ordering
   is n=1 per cell with gaps smaller than between-episode noise. The desk
   corpus stands as pilot data; it is not pooled with rig results.
10. **The rig's first full run (2026-09-02) confirmed H1 on independent
    hardware data** — the VLM's contribution is confined to refusing false
    presence, not to holding belief through occlusion — visible in all three
    places the pilot found it: control/environment ceiling (0/515 frames
    diverging, then 0/18 episodes diverging again on new data), occlusion
    (false belief halves, accuracy flat), and identity (false belief halves,
    accuracy *falls* — see point 2).
11. **A new limitation was found that the pilot corpus could not reveal: the
    VLM is blind to failures the detector never flags.** It is event-gated —
    called only when the pipeline's own evidence looks wrong — so on
    `substitution` (a differently-shaped, differently-coloured object at the
    target's exact position and size) the detector stays confident and the
    VLM is asked 2.0 times per episode against 6.5 on identity, doing almost
    nothing (103 → 100 false-belief frames). This is structural to the
    perception/semantics tier split (§2), not a fixable prompt or threshold.
12. **The decay-timer tradeoff flagged in point 4 is no longer
    unmeasurable.** At 12.1% absent frames (up from the pilot's 6.0%),
    raising `DECAY_FRAMES` costs +39 false-belief frames in total, but that
    cost is concentrated entirely on `unexpected` (+34, patience is wrong
    when the target is truly gone) and nearly absent on `occlusion` (+1,
    patience is correct when it is truly hidden). The fix this earns is a
    **state-conditioned timer**, not a single global constant — long under
    OCCLUDED, short on evidence of substitution — which the belief state
    machine already has the information to compute.

**Next, to close the largest remaining gaps (`LIVE_SESSION_PLAN.md`, ordered
by value):** (1) the ≥40%-absent corpus is still the single highest-value
open item — 12.1% was enough to *show* the timer tradeoff is real and
axis-dependent (point 12), not enough to pin down a safe global value or to
clear the project's own 15%-absence warning; (2) an occluder-diversity pass
through the now-built rig (hand/book/paper/box/jacket at 0/low/high clutter)
— the 2026-09-02 run used one generic occluder throughout, so the
per-occluder ranking (point 8 / §9) is still open even though the rig it
needs now exists; (3) re-running the substitution/identity axes with a
second, independent object pair, since n=6 per axis on one scene supports
the *direction* of points 2 and 11 but not an effect size. Two items from the
previous plan are now closed by the 2026-09-02 run rather than needing
separate action: a valid identity episode (point 2) and the void
`shadow`/`glare` trials, superseded by the rig's `lighting`/`camera_pose`
scenarios, which replicated the ceiling result at n=12 instead of n=1.
Cluttered directional search stays explicitly **blocked** until a direction
mechanism beats 0.50 offline — item 5 in `LIVE_SESSION_PLAN.md` shows why
running it now would only measure a disconnected mechanism; the rig run
did not touch this axis and does not change its status.
