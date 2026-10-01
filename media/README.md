# Evidence set: controlled-rig run `2026-09-01_233533`

Rebuilt from any run with `python make_figures.py runs/<stamp>` (run from `avi/`).

The raw capture is 3.1 GB across 16,856 files and is not in this repository.
This is the curated version, 13 MB, and it is enough to check the claims in
the paper's claims by eye (and `docs/technical-writeup.md`, sections 5c and 5d).

## What is here

`baseline_scan.jpg` is the reference scan taken before any trial, bare rig,
cup on the marked centre position. Every later observation is compared
against it.

`contact_sheets/` holds one sheet per trial, 48 of them, named
`trial_<NNN>_<scenario>_rep<N>_<clutter>.jpg`. Each sheet is 16 frames sampled
evenly across the whole episode. The tiles come from the annotated frames, so
each one already carries the detector box, the belief state, the ground truth
for that phase, and the elapsed time. Frames marked `edge` are labelled as
such and were excluded from grading.

`videos/` holds one clip per scenario at 10 fps, plus four clips of the trials
that carry the findings:

| clip | what it shows |
|---|---|
| `KEY_removal_high_vlm_prevents_ghost_trial014.mp4` | The cup is taken away and the detector keeps firing. Without the VLM this trial records 37 false-belief frames. With it, zero. |
| `KEY_identity_high_vlm_prevents_ghost_trial009.mp4` | The same, on the identity axis: 26 false-belief frames without the VLM, zero with it. |
| `KEY_removal_high_both_fail_trial046.mp4` | Both conditions record 40 false-belief frames. The detector locks onto a substitute mug and neither condition recovers. |
| `KEY_substitution_high_vlm_never_asked_trial031.mp4` | Pink pliers replace the cup. 40 false-belief frames in both conditions and one VLM call in the whole episode, because nothing looked wrong enough to escalate. |

## Reading a contact sheet

The belief state is the word after the phase label. `CONFIRMED` and `OCCLUDED`
both mean the system asserts the target is present; `MISSING`, `DISPLACED` and
`AMBIGUOUS` mean it does not. Compare that against the `truth` value on the
same tile.

A tile where belief says present and truth says ABSENT is a false-belief
frame, which is the error this project treats as the serious one. A tile the
other way round is a lost frame, which costs a wasted search and nothing else.

## Why these images are published

The desk corpus from August is not in this repository and will not be. Those
frames were shot in a bedroom and show an open laptop with readable documents,
a closet, and the operator. The controlled rig is a staged surface holding a
cup, a cardboard box, a pair of pliers and some mugs against a plain
background, so the restriction does not apply to it.
