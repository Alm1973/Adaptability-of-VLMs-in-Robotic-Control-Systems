# Research Question, Hypothesis, and Title (Revised — Assignment 5)

The previous research question referenced "a locally run large language model
on a Raspberry Pi 5." The actual system runs **Moondream, a vision-language
model**, locally via Ollama, currently on a laptop, with **OpenCV handling
spatial perception** in a hybrid pipeline (see
`docs/project-database.md`). This document brings the framing in line with
what's actually being built and tested.

## Revised Research Question

> Can a locally-run vision-language model, operating within a hybrid
> perception-reasoning pipeline where OpenCV handles spatial localization,
> reliably recover from task-disrupting situations — occlusion, environmental
> change, and abnormal/unrecognized targets — during an object verification
> task, by adapting its recovery strategy and repositioning the camera
> autonomously?

Changes from the original:
- "Raspberry Pi 5" → generalized to "locally-run," since the current
  compute host is a laptop (Raspberry Pi 5 remains a target deployment
  platform, not the current one).
- "large language model" → "vision-language model," since Moondream
  processes images directly, unlike a text-only LLM.
- Explicitly names the hybrid OpenCV + VLM architecture in the question
  itself, since the division of labor (OpenCV localizes, VLM reasons about
  recovery) is central to the method, not just an implementation detail.

## Revised Hypothesis

The ranked prediction from the original hypothesis still holds conceptually —
**occlusion easiest to recover from, environmental change moderate, anomaly
hardest** — but the reasoning now explicitly accounts for the hybrid
architecture rather than treating "the model" as a single monolithic system:

> Recovery performance for each failure category depends on two separable
> factors: (1) OpenCV's detection robustness under that failure condition,
> and (2) the VLM's ability to reason about an appropriate recovery action
> once detection degrades or fails. Occlusion is predicted to be the easiest
> failure category to recover from because OpenCV can often still detect a
> partially-visible object above the area threshold, giving the VLM a
> reasonably informative starting signal. Environmental change (e.g. lighting
> shifts) is predicted to be moderate, since it degrades OpenCV's HSV-based
> detection more severely, pushing more of the burden onto the VLM's
> reasoning with less reliable input. Anomalous/unrecognized targets are
> predicted to be hardest, because OpenCV's color-based detection provides
> little to no useful signal for an object outside the calibrated HSV ranges,
> forcing the VLM to reason with minimal perceptual grounding.

## Proposed Working Title

**"AVI: Recovering from Disruption in a Hybrid Vision-Language-Model
Robotic Search System"**

Alternatives considered:
- *A Hybrid OpenCV–VLM Architecture for Adaptive Object Search Under
  Occlusion, Environmental Change, and Anomaly*
- *Adaptability of Vision-Language Models in Robotic Control Systems*
  (current repo/GitHub title — still accurate and reasonable to keep if a
  shorter, more general title is preferred for the README)

Recommendation: keep the existing GitHub repo name/title as-is for
continuity, and use the AVI-specific title above for the paper itself, since
it more precisely names the hybrid architecture that is the actual
contribution.
