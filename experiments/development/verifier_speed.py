import hashlib
import time

import cv2

import disruption_bench as db

PROMPT = "Answer yes or no only. Is the main object in this image a {}?"
WARMUP = 3


def build_crops(idx, target, per_type=6):
    crops = []
    seen = set()

    def add(ep, frame_i, label, tag):
        box = ep["meta"]["box"]
        img = cv2.imread(ep["frames"][frame_i])
        if img is None:
            return
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        h, w = img.shape[:2]
        c = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if not c.size:
            return
        key = hashlib.md5(c.tobytes()).hexdigest()
        if key in seen:
            return
        seen.add(key)
        crops.append((c, label, tag))

    VARYING = ("lighting", "white_balance", "shadow", "glare",
               "occlusion_partial")
    for name, ep in idx.items():
        t = ep["meta"]["type"]
        d0, rec = ep["meta"]["disrupt_start"], ep["meta"]["recover_start"]
        add(ep, 0, True, "clean target")
        if t in VARYING:
            for i in range(d0, min(rec, d0 + per_type)):
                add(ep, i, True,
                    "partly occluded" if t == "occlusion_partial"
                    else "target under " + t)

    for name, ep in idx.items():
        t = ep["meta"]["type"]
        n, rec = ep["n"], ep["meta"]["recover_start"]
        tag = {"impostor_same_class": "impostor",
               "novel_at_target": "replaced",
               "occlusion_remove": "empty desk"}.get(t)
        if tag:
            for i in range(rec, min(n, rec + per_type)):
                add(ep, i, False, tag)
    return crops


def bench(name, fn, crops, prompt):
    for c, _, _ in crops[:WARMUP]:
        fn(prompt, c)

    times, answers = [], []
    for c, _, _ in crops:
        t0 = time.time()
        a = fn(prompt, c)
        times.append(time.time() - t0)
        answers.append(str(a).strip().lower().startswith("y"))
    mean = sum(times) / len(times)
    correct = sum(a == lbl for a, (_, lbl, _) in zip(answers, crops))
    print(f"  {name:<12} mean {mean:.3f}s  median "
          f"{sorted(times)[len(times) // 2]:.3f}s  "
          f"acc {correct}/{len(crops)} ({correct / len(crops):.3f})")
    return {"mean": mean, "answers": answers, "correct": correct}


def main():
    from run_study import make_verifier, make_fast_verifier

    idx = db.load()
    target = {e["meta"]["target"] for e in idx.values()}.pop()
    crops = build_crops(idx, target)
    if not crops:
        raise SystemExit("ABORT: no crops built -- regenerate episodes")
    prompt = PROMPT.format(target)
    pos = sum(1 for _, l, _ in crops if l)
    majority = max(pos, len(crops) - pos) / len(crops)
    print(f"{len(crops)} labelled crops ({pos} yes / {len(crops) - pos} no), "
          f"target={target!r}")
    print(f"  majority-class baseline: {majority:.3f} "
          f"(always answering {'yes' if pos > len(crops) - pos else 'no'}) "
          f"-- beat this or the verifier is doing nothing")
    by_tag = {}
    for _, lbl, tag in crops:
        by_tag[tag] = by_tag.get(tag, 0) + 1
    print(f"  {by_tag}\n")

    import gc
    import torch

    print("loading generate verifier...")
    gen = make_verifier()
    print("\nLATENCY AND ACCURACY")
    r_gen = bench("generate_8", gen, crops, prompt)
    r_gen2 = bench("generate_2", lambda p, c: gen(p, c, tokens=2),
                   crops, prompt)

    del gen
    gc.collect()
    torch.cuda.empty_cache()

    print("loading logits verifier...")
    fast = make_fast_verifier()
    r_fast = bench("logits", fast, crops, prompt)

    agree = sum(a == b for a, b in zip(r_gen["answers"], r_fast["answers"]))
    n = len(crops)
    speedup = r_gen["mean"] / r_fast["mean"] if r_fast["mean"] else 0

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    agree2 = sum(a == b for a, b in zip(r_gen["answers"], r_gen2["answers"]))
    print(f"  speedup      logits {speedup:.2f}x  "
          f"({r_gen['mean']:.3f}s -> {r_fast['mean']:.3f}s)")
    print(f"               generate_2 "
          f"{r_gen['mean'] / r_gen2['mean'] if r_gen2['mean'] else 0:.2f}x  "
          f"({r_gen['mean']:.3f}s -> {r_gen2['mean']:.3f}s), "
          f"agreement {agree2}/{n}")
    print(f"  agreement    {agree}/{n} ({agree / n:.3f}) with the shipping path")
    print(f"  accuracy     generate {r_gen['correct']}/{n}   "
          f"generate_2 {r_gen2['correct']}/{n}   "
          f"logits {r_fast['correct']}/{n}")

    diffs = {}
    for (a, b), (_, lbl, tag) in zip(zip(r_gen["answers"], r_fast["answers"]),
                                     crops):
        if a != b:
            diffs[tag] = diffs.get(tag, 0) + 1
    if diffs:
        print(f"  disagreements by case: {diffs}")

    print()
    if agree / n >= 0.95 and r_fast["correct"] >= r_gen["correct"]:
        print("  ADOPT: same answers, at least as accurate, and faster. Prior "
              "study\n  results remain valid because the verifier's behaviour "
              "is unchanged.")
    elif agree / n >= 0.95:
        print("  ADOPT WITH CARE: answers agree but accuracy is slightly "
              "lower. Worth it\n  only if the latency matters more than the "
              "lost points -- decide against\n  the live loop, not in the "
              "abstract.")
    else:
        print("  DO NOT ADOPT: the fast path answers differently often enough "
              "that it is\n  a different verifier. Adopting it would silently "
              "invalidate the ablation\n  results, which is a far worse "
              "outcome than a slow call.")


if __name__ == "__main__":
    main()
