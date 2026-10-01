# Multi-Modal Detection Tools — Architecture Design

Design/sketch only. Nothing in this document is wired into `tracker.py`,
`config.py`, `vlm_recovery.py`, or `main.py`. Function signatures below
are proposals, not implementations.

## The question

Should classical-CV metrics (convexity defects, ellipse fit, specular
highlight, etc.) be:

**(A) Computed and fed to the VLM as text context**, appended to the
existing `VERIFICATION_PROMPT` the same way `scan_history` and
`last_known_direction` are already appended to the REACQUIRE prompt —
i.e. the VLM still makes the final call, now with extra numeric hints.

**(B) Selectable/switchable tools the VLM picks between itself**,
matching Assignment 3's original framing (the VLM choosing which
detection tool to invoke).

## Recommendation: neither alone — a 3-tier hybrid, and here's why

Both A and B ask the VLM to do something this project has repeatedly
found `qwen2.5vl:3b` struggles with. Option A asks it to correctly weigh
several numeric values against visual judgment in one pass — a milder
version of the same "precise structured reasoning" task that failed
in the mosaic-coordinate work (`target: (row, col)` had to be abandoned
for a coarser left/right/up/down ask, twice, at two different grid
resolutions, per `PROGRESS.md`'s earlier session). Option B asks it to
*select among abstract tool names* based on visual judgment about which
property is most diagnostic right now — that's an extra layer of abstract
reasoning on top of the classification task it's already doing, and
this project's track record with this specific model is that added
abstraction layers tend to degrade reliability, not improve it.

**Better fit for what's actually been learned in this project: let
classical CV do a fast, cheap, deterministic pre-filter, and only call
the VLM when that pre-filter is inconclusive.** This doesn't ask the
model to do anything harder than it already does today (classify one
image) — it just sometimes skips asking it at all, when a strong
classical signal already exists. It also plays to each component's
actual strength: classical CV is fast/free/deterministic where it's
confident, the VLM is used for genuinely ambiguous cases where whole-image
visual judgment is actually needed (which is what it's good at, per
`verify_test.py`'s clean 2/2 result on unambiguous cup images).

This is arguably *more* faithful to Assignment 3's "VLM chooses between
tools" spirit than option B, just implemented as a confidence gate rather
than an explicit VLM tool-selection call — the "choice" is still there,
it's just made by whether the classical signal crosses a confidence
threshold, not by asking the model to name a tool.

## Sketch: data flow

```
find_red() [tracker.py, already returns box/center/area]
        |
        v
compute_shape_signals(frame, contour, box) -> dict
        |
        v
classify_from_signals(signals) -> "cup" | "other" | None
        |
   None?|  "cup"/"other"?
        |         |
        v         v
  get_verification()   return directly, no VLM call
  [existing, unchanged]
        |
        v
   "cup" | "other" | None (degenerate, existing handling unchanged)
```

## Sketch: function signatures

```python
# Would live in a new module, e.g. shape_signals.py, OR as additions to
# vlm_recovery.py -- not decided, not implemented.

def compute_shape_signals(frame, contour, box) -> dict:
    """Runs all lightweight classical-CV metrics once. All are sub-ms to
    low-ms on Pi5-class hardware (see Task 1 research) -- cheap enough to
    always compute regardless of which downstream path gets used.

    IMPORTANT per Task 2's finding: any per-pixel metric here (e.g.
    specular highlight) MUST mask to the actual contour, not the
    bounding box -- the box-based version was proven wrong in testing
    today (see PROGRESS.md Task 2 section)."""
    return {
        "convexity_defects": count_significant_convexity_defects(contour),
        "ellipse_fit_quality": ellipse_fit_quality(contour),
        "solidity": ...,  # cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour))
        "specular_highlight": specular_highlight_ratio_masked(frame, contour, box),
    }


def classify_from_signals(signals: dict) -> str | None:
    """Fast deterministic gate. Returns "cup" or "other" only when
    confident; None means inconclusive -> caller falls through to the
    VLM. Thresholds below are PLACEHOLDERS -- not tuned, pending real
    hand-only test data (see completion summary). Do not treat these
    numbers as validated."""
    if signals["convexity_defects"] >= 2:
        return "other"  # confident: finger-gap signature present
    # Deliberately not adding more confident-only rules yet -- every
    # metric tested today was either inconclusive or confounded by the
    # ground-truth problem. Only convexity_defects has real literature
    # support (Task 1) independent of today's specific test images.
    return None


def get_verification_hybrid(frame, contour, box) -> str | None:
    """Proposed new entry point for vlm_recovery.py (NOT implemented).
    Tries the fast classical-CV gate first; only pays the ~1s+ VLM round
    trip when classical signals are inconclusive. Return contract
    (cup/other/None) matches the existing get_verification() exactly, so
    main.py's calling code wouldn't need to change at all -- only the
    function being called would."""
    signals = compute_shape_signals(frame, contour, box)
    fast_result = classify_from_signals(signals)
    if fast_result is not None:
        print(f"[VERIFY] Fast classical-CV classification: {fast_result} "
              f"(signals={signals})")
        return fast_result

    print(f"[VERIFY] Classical-CV inconclusive (signals={signals}), "
          f"falling back to VLM")
    return get_verification(frame, box)  # existing function, unchanged
```

## What would actually need to change in live files (future work, not done)

- `tracker.py`: `find_red()` would need to also return the raw `contour`
  object in its result dict, not just `box`/`center`/`area` -- currently
  the contour itself is discarded after `boundingRect()` is called.
- `main.py`: `maybe_trigger_verification()` and `verification_worker()`
  would need to thread `contour` through alongside `box`/`center`.
- `vlm_recovery.py`: would gain the new functions above; `main.py`'s
  import would change from `get_verification` to
  `get_verification_hybrid` (or `get_verification_hybrid` could just
  replace `get_verification`'s internals directly, keeping the name and
  call site identical -- smaller diff, arguably cleaner).

None of this is being done now -- this is scoped as a future
implementation task once real hand-only test data exists to validate
`classify_from_signals()`'s thresholds against.

## Alternative sketch: Option B (VLM tool-selection), for completeness

Sketched since the task asked for it, even though not recommended above:

```python
TOOL_DESCRIPTIONS = {
    "shape_check": "Analyzes the object's outline for finger-like gaps",
    "reflectance_check": "Checks for glossy highlights typical of plastic",
}

# Round 1: VLM picks a tool by name (same short-word-response pattern as
# the rest of this project's prompts)
# Round 2: run the selected tool, feed its result + the image back to the
# VLM for a final classification

# Two VLM round trips instead of a variable 0-or-1 -- strictly more
# latency than the hybrid approach in the common case, and asks the model
# to do the abstract "which tool is relevant" judgment call this project
# has repeatedly found unreliable at this model size. Not recommended,
# but sketched per the task's request.
```
