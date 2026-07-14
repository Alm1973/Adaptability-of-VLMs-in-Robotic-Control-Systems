# VLM Spatial Reasoning: Literature Review & Division of Labor

Assignment 3. Covers: (a) what current research says about VLM spatial reasoning capability, (b) a template for testing Moondream directly, (c) the resulting division of labor between OpenCV and the VLM in AVI's pipeline.

## (a) What the literature says

**Basic left/right/near/far discrimination is largely a solved, saturating task for large frontier VLMs — but not for small/edge models, and not for anything beyond the basics.**

- Benchmarks like SpatialBot-Bench and EmbSpatial show large frontier models (o3, Gemini-2.5-Pro) scoring above 90% on simple position relationships (left/right/front/back) and proximity (near/far), which the OmniSpatial benchmark authors describe as an approaching-saturation task for that model tier.
- However, more targeted benchmarks tell a less rosy story even for capable models. LRR-Bench found that while some large open models (e.g. Qwen2.5-72B) lead on basic position tasks, all tested models struggle significantly once the task involves object movement or 3D/rotation reasoning rather than a static left/right label.
- A related finding worth noting for the project: LRR-Bench's authors report that reasoning techniques like chain-of-thought don't reliably improve spatial understanding, and that simply scaling model parameters doesn't fix it either — spatial reasoning behaves differently from general language reasoning benchmarks.
- Earlier work (the ARO benchmark) found VLMs often behave like "bag-of-words" models with weak relational understanding and word-order insensitivity, meaning a model may recognize *that* two objects and a spatial word are all present without correctly resolving *which* object the relation applies to.

**On small/edge models specifically (most relevant to AVI, since Moondream is the model in use):**

- Moondream is explicitly designed and marketed as a small (0.5B–2B parameter class), edge-deployable VLM, positioned to run on CPUs, mobile devices, and Raspberry Pis rather than datacenter GPUs — the tradeoff for that efficiency is generally reduced capability relative to frontier-scale models on harder reasoning benchmarks, even if basic tasks remain feasible.
- General findings on small/efficient VLMs (per a broader efficiency survey) show comparable lightweight models achieving competitive results on basic multimodal benchmarks while keeping latency low, which is consistent with what the project's own benchmarking already found (Moondream fastest at ~4.8s vs. ~68s for a heavier CPU-based LLaVA baseline) — but that survey doesn't specifically isolate spatial-reasoning accuracy at this model scale, which is exactly the gap AVI's own Assignment 3(b) test is meant to probe empirically rather than assume from the literature.

**Takeaway for AVI's methodology:** the literature supports what the project has already found empirically — spatial reasoning is a known weak point for VLMs in general, and is a bigger risk at Moondream's small model scale specifically. This is direct justification for why AVI does **not** ask the VLM to infer pixel coordinates from raw images, and instead uses OpenCV to do the actual spatial localization, handing the VLM only a simple categorical label to reason over.

**Sources:**
- OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models (arXiv 2506.03135)
- LRR-Bench: Left, Right or Rotate? Vision-Language Models Still Struggle With Spatial Understanding Tasks (arXiv 2507.20174)
- Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models (OpenReview)
- Moondream model documentation (moondream.ai/p/models)
- A Survey on Efficient Vision-Language Models (arXiv 2504.09724)

## (b) Test Moondream directly — protocol + script


**Protocol:**
1. Capture 3–5 still frames with a single object at different, known positions (e.g. left, center, right).
2. For each frame, ask Moondream the three questions below and record its literal answer next to the true answer.
3. Compute simple accuracy (correct / total) per question type.

```python


import ollama

QUESTIONS = [
    "Where in the frame is the {color} object - left, center, or right?",
    "Is the {color} object closer to the top or bottom of the image?",
]

TWO_OBJECT_QUESTION = (
    "There are two objects in the image, a {color_a} one and a {color_b} one. "
    "Which one is further to the left?"
)


def probe_single(image_path, color, true_answer_left_right, true_answer_top_bottom):
    results = {}
    for template in QUESTIONS:
        q = template.format(color=color)
        response = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": q, "images": [image_path]}],
        )
        results[q] = response["message"]["content"].strip()
    print(f"--- {image_path} ---")
    for q, a in results.items():
        print(f"Q: {q}\nA: {a}")
    print(f"(Expected: left/right={true_answer_left_right}, "
          f"top/bottom={true_answer_top_bottom})\n")
    return results


if __name__ == "__main__":
    pass
```

**Results table :**

| Frame | Question | Moondream Answer | Ground Truth | Correct? |
|---|---|---|---|---|
| red_left.jpg | left/center/right | | left | |
| red_center.jpg | left/center/right | | center | |
| red_right.jpg | left/center/right | | right | |
| (two-object frame) | which is further left | | | |




