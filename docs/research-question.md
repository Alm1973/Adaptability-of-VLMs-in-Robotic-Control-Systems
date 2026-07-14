# Research Question, Hypothesis, and Title


## Revised Research Question

> Can a local vision-language model, integrated in a hybrid perception-reasoning pipeline where OpenCV is employed >
> for spatial localization, robustly recover from task-disrupting scenarios – occlusion, environmental change, and >
> abnormal/unrecognized targets – in an object verification task, through adaptation of the recovery approach and >
> autonomous repositioning of the camera?


- "Raspberry Pi 5" → generalization to "locally-run," because the
  current compute host is a laptop ("Raspberry Pi 5" is only a target for
  deployment, not the current host).
- "large language model" → "vision-language model," since Moondream
  operates on images directly, and is therefore not an LLM in the strict sense.
- Explicit mention of hybrid OpenCV+VLM architecture in the problem
  statement, because the specialization of tasks (OpenCV does localization,
  VLM does reasoning about recovery) is important to the solution.

## Revised Hypothesis

However, while the ordered prediction of the original hypothesis remains
theoretically accurate, that reasoning now incorporates the architecture's
hybrid nature:
> Performance of recovery from each failure type will depend on two separate
> factors: (1) Robustness of OpenCV detection in that situation, and (2)
> VLM's ability to reason appropriately about a recovery action based on the
> deteriorating/delimited detection. Occulsion is expected to be the easiest
> failure type to recover from, because OpenCV will usually still be able to
> detect at least some portion of an occluded object above the area
> threshold, providing the VLM with a fairly clear signal. Environmental
> change (e.g. changing lighting) is expected to be moderately difficult,
> because such changes degrade HSV-based OpenCV detection, shifting the
> burden more onto the VLM's reasoning from less reliable signals. Anomalies/
> unrecognized objects are expected to be the most difficult type of failure
> to recover from, because OpenCV's color detection offers little/no usable
> signal for any object outside the trained HSV ranges.
>
