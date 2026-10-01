import json
import re
import statistics as st
import time

import cv2

import llm_backend
from heldout_gt import HALLUCINATION_PROBES, HELDOUT_GT, class_balance
from open_vocab_detect import LIST_PROMPT, _dedupe, _parse_items, is_present

N_SAMPLES = 3
REPEATS = 3


def sample_lists(img, n):
    out = []
    for _ in range(n):
        txt = llm_backend.vision_chat(LIST_PROMPT, img, max_tokens=140)
        out.append(_dedupe(_parse_items(txt)))
    return out


def decide(obj, lists, mode):
    hits = sum(1 for l in lists if is_present(obj, l))
    if mode == "single":
        return 1 if hits >= 1 else 0
    if mode == "union3":
        return 1 if hits >= 1 else 0
    if mode == "major3":
        return 1 if hits >= 2 else 0
    raise ValueError(mode)


def probe_hits(lists, mode):
    n = 0
    for p, syns in HALLUCINATION_PROBES.items():
        c = sum(1 for l in lists
                if any(re.search(rf"\b{re.escape(s)}\b", it.strip().lower())
                       for it in l for s in syns))
        if c >= (2 if mode == "major3" else 1):
            n += 1
    return n


def evaluate(all_lists, mode):
    tp = tn = fp = fn = 0
    halluc = 0
    for f, labels in HELDOUT_GT.items():
        ls = all_lists[f][:1] if mode == "single" else all_lists[f]
        halluc += probe_hits(ls, mode)
        for obj, gt in labels.items():
            pred = decide(obj, ls, mode)
            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 0:
                tn += 1
            elif gt == 0 and pred == 1:
                fp += 1
            else:
                fn += 1
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return ((tp + tn) / n if n else None, sens, spec, halluc, n)


def main():
    print("HELD-OUT class balance:")
    for obj, (p, n) in class_balance().items():
        print(f"  {obj:<15} present {p}  absent {n}")
    frames = list(HELDOUT_GT)
    imgs = {f: cv2.imread(f) for f in frames}
    missing = [f for f, im in imgs.items() if im is None]
    if missing:
        print("MISSING FRAMES:", missing)
        return
    print(f"\n{len(frames)} frames x {N_SAMPLES} samples x {REPEATS} repeats\n")

    runs = {"single": [], "union3": [], "major3": []}
    for rep in range(1, REPEATS + 1):
        t0 = time.time()
        all_lists = {f: sample_lists(imgs[f], N_SAMPLES) for f in frames}
        dt = time.time() - t0
        print(f"repeat {rep}: sampled in {dt:.0f}s")
        for mode in runs:
            acc, sens, spec, hal, n = evaluate(all_lists, mode)
            runs[mode].append({"acc": round(acc, 3), "sens": sens,
                               "spec": spec, "halluc": hal, "n": n})
            print(f"  {mode:<8} acc {acc:.3f} ({int(acc*n)}/{n})  "
                  f"sens {sens}  spec {spec}  halluc {hal}")
        json.dump(runs, open("vote_decision.json", "w"), indent=2)
        print()

    llm_backend.shutdown()
    print("===== DECISION =====")
    print(f"{'mode':<8} {'mean_acc':<10} {'spread':<9} {'mean_sens':<11} "
          f"{'mean_spec':<11} {'halluc'}")
    means = {}
    for mode, rs in runs.items():
        accs = [r["acc"] for r in rs]
        sens = [r["sens"] for r in rs if r["sens"] is not None]
        spec = [r["spec"] for r in rs if r["spec"] is not None]
        hal = sum(r["halluc"] for r in rs)
        means[mode] = st.mean(accs)
        print(f"{mode:<8} {st.mean(accs):<10.3f} "
              f"{max(accs)-min(accs):<9.3f} "
              f"{st.mean(sens):<11.3f} {st.mean(spec):<11.3f} {hal}")
    best = max(means, key=means.get)
    print(f"\nBest mean held-out accuracy: {best} ({means[best]:.3f})")
    print("SHIP RULE: adopt union3 only if it beats single on mean held-out "
          "accuracy AND total hallucinations stay 0.")


if __name__ == "__main__":
    main()
