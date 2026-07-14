# Grid Testing Protocol (Assignment 2)

Purpose: prove spatial detection accuracy holds across the frame, not just at
the 3 positions already tested (center, top-left, bottom-right). Camera lens
distortion, mounting angle, and lighting are not symmetric across the frame,
so performance at one position doesn't guarantee performance at another.

**This requires physical trials with your actual camera and objects — I can't
generate these results. Run the protocol below and fill in the table.**

## Protocol

For each of the 5 grid positions — **center, top-left, top-right, bottom-left,
bottom-right** — run the same 3-round test already used for the original
center test:

1. **Centered** — object placed at the nominal center of that grid square.
2. **Shifted-left** — object shifted toward the left edge of that square.
3. **Shifted-right** — object shifted toward the right edge of that square.

That's 5 positions × 3 rounds = 15 trials minimum; the mentor's assignment asks
for 9 trials/position in some framings — run 3 repetitions of each round if you
want the full 45-trial version for stronger statistics.

For each trial:
- Place the object.
- Run `code/detection.py` (or your full pipeline) and record the grid label it outputs.
- Mark **correct** if the label matches where you actually placed the object, **incorrect** otherwise.
- If incorrect, immediately log why in `lab-notebook.md` per Assignment 2(d) — don't wait, you won't remember later.

## Results table

Fill this in as you go. Copy additional rows if you run more than 3 repetitions per round.

| Grid Position | Round Type | Correct / Total | Accuracy | Notes |
|---|---|---|---|---|
| Center | Centered | ? / 3 | ? | |
| Center | Shifted-left | ? / 3 | ? | |
| Center | Shifted-right | ? / 3 | ? | |
| Top-left | Centered | ? / 3 | ? | |
| Top-left | Shifted-left | ? / 3 | ? | |
| Top-left | Shifted-right | ? / 3 | ? | |
| Top-right | Centered | ? / 3 | ? | |
| Top-right | Shifted-left | ? / 3 | ? | |
| Top-right | Shifted-right | ? / 3 | ? | |
| Bottom-left | Centered | ? / 3 | ? | |
| Bottom-left | Shifted-left | ? / 3 | ? | |
| Bottom-left | Shifted-right | ? / 3 | ? | |
| Bottom-right | Centered | ? / 3 | ? | |
| Bottom-right | Shifted-left | ? / 3 | ? | |
| Bottom-right | Shifted-right | ? / 3 | ? | |

**Overall accuracy:** ? / ? = ?%

## Pattern analysis (fill in after collecting data)

Questions to answer once the table is complete, per Assignment 2(c):
- Is accuracy worse at the edges/corners than at center?
- Is one corner consistently worse than the others? (Could indicate uneven lighting or lens distortion concentrated on one side.)
- Are "shifted" rounds near a grid-cell boundary more error-prone than centered rounds? (Expected — boundary cases are inherently ambiguous for a 3×3 discretization.)

_Write 2–3 sentences here once you have real numbers — this becomes real data for the paper._

## Failure log

For every incorrect trial, per Assignment 2(d) — don't skip this even if it feels obvious in the moment:

| Trial | Grid Position | System Output | What You Think Went Wrong |
|---|---|---|---|
| | | | |
