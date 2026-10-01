# Live session plan — everything left that needs hardware

Written 28 Aug 2026, after Assignments 1–4 were completed offline.
**Revised same day:** all remaining captures move to a controlled rig. See
*Decision: controlled environment* below — it changes where these run, not
what they test.

**Status update, 2026-09-02, after the rig's first full run
(`runs/2026-09-01_233533`, see `PROGRESS.md`'s 2026-09-02 entry and
`WRITEUP.md` §5c).** The rig now exists and one attended session has run
through it: 48 trials, 8 scenarios x 3 reps x 2 clutter levels. This closed
or partially closed several items below:
- **Item 2 (identity episode) is DONE** — `identity_swap` is a valid episode
  and reproduces the synthetic identity finding live. No further identity
  capture is needed to answer the question as originally posed; a *second*
  independent object pair would strengthen the effect-size estimate (n=6
  currently) but is not required to close the gap.
- **Item 4 (shadow/glare re-runs) is SUPERSEDED**, not literally done — the
  rig's `lighting` and `camera_pose` scenarios replicate the same ceiling
  result (D and E identical, 0 false belief) at n=12 instead of the old void
  n=1 trials. The specific `shadow.bat`/`glare.bat` void trials do not need
  re-running.
- **Item 3 (clutter_similar re-capture) is SUPERSEDED** — clutter is now a
  factorial variable (none/high) applied to every scenario in the rig, which
  is exactly what this item asked for ("captured as part of the grid rather
  than separately"). The old 136 orphan frames remain unscored and are not
  worth resurrecting now that fresh, GT-tagged clutter data exists.
- **Item 1 (absence-heavy corpus) is PARTIALLY addressed, still the top open
  item.** The run reached 12.1% absent frames (691/5,699), up from the
  pilot's 6.0%, and that was enough to show the decay-timer tradeoff is real
  and axis-dependent (§5c of `WRITEUP.md`). It is still short of both the
  ≥40% target below and the project's own 15%-absence warning threshold, so
  a dedicated absence-heavy capture remains worth doing — now to sharpen a
  confirmed tradeoff, not to make an unmeasurable one measurable.
- **New open item, not in the original plan:** the occluder used throughout
  this run was a single generic object (a cardboard box), so the
  occluder-diversity factorial in the rig spec below (hand/book/paper/box/
  jacket at 0/low/high clutter) has still never been run *in the rig*,
  despite the rig now existing. This is what item 1 in "Order of operations"
  below now refers to, alongside the absence-heavy capture.
- **Item 5 is untouched** — the run's 8 scenarios do not include a search
  task, so nothing here changes its BLOCKED status.

**Hardware rule, unchanged:** nothing in this file runs unattended. A human is
physically present and watching for anything that moves the arm.

Items 1–3 are **camera only — the arm does not move.** Item 4 moves the arm.
Item 5 is blocked and must not be run yet.

Ordered by how much each one unblocks.

---

## Decision: controlled environment

**All remaining captures move to a controlled rig with a plain background.
Clutter becomes a manipulated variable instead of a nuisance variable.**

**Why.** The desk carries too many uncontrolled variables to attribute a
result to a cause. Two findings from 28 Aug make this concrete:

- Assignment 3(a): D and E disagree on **0 of 515 frames** on the environment
  and unexpected axes — both pinned at 1.000. The corpus cannot discriminate.
  In a controlled rig, clutter is dialled up until conditions separate.
- Assignment 4, conclusion 11: the per-occluder ordering (book 0.562 > jacket
  0.429 > box 0.387 > paper 0.000) is marked **SUGGESTIVE, do not rank** —
  gaps smaller than between-episode variation, every cell n=1. With the
  occluder as the only thing changing, that ordering becomes rankable.

**The desk corpus is not discarded.** It is reframed as pilot/exploratory data
that generated the hypotheses. The controlled rig tests them. Cross-corpus
numbers are not pooled.

### What does NOT need re-running (environment-independent)

| result | why it transfers |
|---|---|
| Assignment 2, all four tests | deterministic re-scoring of fixed frames |
| `search_direction()` returns `None` | a code fact, not a scene fact |
| Timing profile | hardware, not scene |
| Search reach = 24° | arithmetic |

### What MUST be re-run

Everything marked SUGGESTIVE in Assignment 4(a): per-occluder ordering, every
n=1 cell, false belief, and the identity episode.

### Rig specification — fix before the first capture

- **Camera distance marked and fixed.** `degrade_bench` puts the detection
  floor at 0.12× resolution; keep apparent cup size well inside that.
- **Lighting fixed and measured**, not ambient room light.
- **Target position marked**, so removal and swap land in the same spot.
- **Clutter as discrete levels** — 0 / low / high, with a written object count
  and list, so "high" means the same thing in week 3 as in week 1.
- **Same occluder set at every level** (hand, book, paper, box, jacket), so
  occluder × clutter is a real factorial.

### Calibration step, before any condition runs

A red cup on a plain background is much higher contrast than a red cup on the
desk. **Capture and score a clean baseline in the new rig first.** Without it,
a clutter effect cannot be told apart from a contrast effect.

---

## 1. Absence-heavy corpus — **highest value** (PARTIAL, see status update)

**Why.** Assignment 2 recommends raising `DECAY_FRAMES` from 12 to 60, which
scores 0.697 against the region probe's 0.504. That recommendation is not safe
to adopt, because false belief stayed at exactly **5 for every timer value
including infinite**. Infinite patience means never conceding the cup is gone
and should generate false belief constantly. It does not, because only **6.0%
of scored frames (66 of 1,099)** have the cup genuinely absent.

The one error that matters for safety is currently almost unmeasurable. A
system that answered "present" on every frame would score 94%.

**What to stage.** 6–8 episodes, each: baseline with the cup → remove it
entirely from the scene → keep recording with it gone for the majority of the
episode. Vary the ending:

| # | ending | what it separates |
|---|---|---|
| 1–2 | cup removed, nothing put back | plain absence |
| 3–4 | cup removed, a *different* object left in its place | absence vs substitution |
| 5–6 | cup removed while a hand is still in frame | can it tell "hidden" from "taken"? |
| 7–8 | cup removed, then returned late | recovery after a long true absence |

**Target:** at least 40% of scored frames absent, against today's 6.0%.

**Run this at HIGH clutter in the controlled rig.** On a bare background a
removed cup leaves uniform emptiness and MISSING becomes trivially
detectable, which is the easy version. Staging removal against dense clutter
is the version that actually tests hidden-vs-gone.

**What it settles.** Whether timer=60 is safe, or whether the region probe
earns its cost after all on data that can actually punish patience. Either
result is publishable and neither is currently knowable.

---

## 2. A valid identity episode — **fourth attempt** (DONE 2026-09-02, see status update)

**Why.** Rejecting same-class impostors is the *only* capability the ablation
attributes to the VLM, and the live corpus contains **zero** episodes that
test it. Three attempts have failed:

| date | what went wrong |
|---|---|
| 13 Aug | GT wrong — cup relocated in-frame, not removed |
| 15 Aug | `impostor_colour` recover phase never swapped the cup |
| 15 Aug | `identity_same_class` — operator removed the cup instead of swapping it |

All three produced a *removal* episode, which is why `KIND` files them under
occlusion.

**What makes it valid — the one thing to get right.** A second cup, **same
size and shape, different colour**, placed at the *same position* as the
first. The target must leave and the impostor must arrive. If at any point
there is no cup in the frame, it is a removal episode again.

Checklist before recording: two cups in hand, same silhouette, different
colour; swap performed in one motion; the frame is never empty.

---

## 3. Re-capture `clutter_similar` (SUPERSEDED, see status update)

**Why.** 136 frames are on disk with all three phases, and **no entry in
`live_disruption.json`** — so no capture-time ground truth. This project does
not assign GT by reviewing footage afterwards, so those frames cannot be
scored for belief and are currently dead weight.

**What to do.** Re-run through `live_disruption.py` so the manifest is written
at capture time. In the new rig this is no longer a one-off scenario — it
becomes the `high` clutter level of the factorial, so it is captured as part
of the grid rather than separately.

---

## 4. Re-run `shadow` and `glare` — arm moves (SUPERSEDED, see status update)

**Why.** Both scored 0/3 and are marked **VOID** in the notebook, not failed.
`shadow_0.jpg` shows a desk with a **navy blue mug and no red cup**;
`glare_0.jpg` shows the arm pointed at the laptop screen. The trials measured
staging errors.

**What to do.** Verify the red cup is in frame and detected *before* starting
each trial, then run 3 trials each.

```bash
nano.bat shadow 3
```

```bash
nano.bat glare 3
```

---

## 5. Cluttered directional search — **BLOCKED, do not run**

**Why not.** Assignment 3(b) asks whether the VLM's directional-search
advantage amplifies in a disordered scene. There is no advantage to amplify:

- `search_direction()` asks a four-way question of `make_fast_verifier()`, a
  yes/no logit comparator. It returns `'no'`, no direction parses, and the
  function returns `None` — **on every frame of every recorded run**. The VLM
  has never steered a search, nano trials included.
- Wired to `.generate` it answers `left` on 80 of 80 real frames.
- Across six prompt formulations and 240 balanced trials, nothing beats
  **0.55** against a **0.50** baseline. One formulation scores **0.075** by
  echoing a direction word out of the prompt.

Running the live version would book an evening to measure a disconnected
mechanism that outputs noise when connected.

**Unblock condition:** a direction mechanism that beats 0.50 on
`search_prompt_sweep.py`. That is a modelling problem — a different model, a
spatial-reasoning head, or dropping the idea and using geometric search
honestly. Only after that does the cluttered scene become worth staging.

---

## Not needed from you

Assignments 1, 2, 3(a), 3(b) and 4 are complete without hardware. Everything
above is additional evidence, not a gap in the assignments as set.

## Order of operations in the new rig

1. ✅ Build the rig and write down the spec above as actual numbers. Done —
   `runs/2026-09-01_233533`, 8 scenarios x 3 reps x 2 clutter levels.
2. ✅ Clean baseline capture + score — the calibration step. Done (`baseline/`
   in the same run).
3. ✅ Identity episode (item 2), with the two-cup checklist. Done —
   `identity_swap`, n=6, reproduces the synthetic finding live.
4. ~~Occluder x clutter factorial (items 3–4 absorbed into this).~~
   **Partially done**: clutter (none/high) ran as a factor on every scenario,
   but the occluder itself was a single generic object (cardboard box)
   throughout, so the occluder-diversity half of this item (hand/book/paper/
   box/jacket) has still never run in the rig. **This is now the next thing
   to stage**, alongside item 5 below.
5. Absence-heavy corpus at high clutter (item 1) — still open, 12.1%
   achieved against a ≥40% target.
6. Directional search stays blocked until it beats chance offline.
