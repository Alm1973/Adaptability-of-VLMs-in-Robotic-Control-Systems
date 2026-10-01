import json
import statistics as st
import time

import cv2

import llm_backend
from heldout_gt import HELDOUT_GT
from open_vocab_detect import (LIST_PROMPT, MAX_TOKENS, _dedupe, _parse_items,
                               is_present)

REPEATS = 3
QUERIES = ["laptop", "keyboard", "computer mouse", "water bottle"]


def one_sample(img):
    txt = llm_backend.vision_chat(LIST_PROMPT, img, max_tokens=MAX_TOKENS)
    return _dedupe(_parse_items(txt)) if txt else []


def votes(queries, lists):
    return {q: sum(1 for l in lists if is_present(q, l)) for q in queries}


def run_fixed3(img, queries):
    lists = [one_sample(img) for _ in range(3)]
    v = votes(queries, lists)
    return {q: v[q] >= 2 for q in queries}, 3


def run_adaptive(img, queries):
    lists = [one_sample(img), one_sample(img)]
    v = votes(queries, lists)
    if all(c in (0, 2) for c in v.values()):
        return {q: v[q] >= 2 for q in queries}, 2
    lists.append(one_sample(img))
    v = votes(queries, lists)
    return {q: v[q] >= 2 for q in queries}, 3


def run_single(img, queries):
    l = one_sample(img)
    return {q: is_present(q, l) for q in queries}, 1


STRATEGIES = {"single": run_single, "fixed3": run_fixed3,
              "adaptive": run_adaptive}


def main():
    frames = list(HELDOUT_GT)
    imgs = {f: cv2.imread(f) for f in frames}
    if any(v is None for v in imgs.values()):
        print("missing frames")
        return
    print(f"ADAPTIVE SAMPLING: {len(frames)} held-out frames x {REPEATS} "
          f"repeats\n")

    results = {k: [] for k in STRATEGIES}
    for rep in range(1, REPEATS + 1):
        print(f"--- repeat {rep} ---")
        for name, fn in STRATEGIES.items():
            correct = total = calls = 0
            t0 = time.time()
            for f in frames:
                preds, n = fn(imgs[f], QUERIES)
                calls += n
                for q in QUERIES:
                    total += 1
                    if int(preds[q]) == HELDOUT_GT[f][q]:
                        correct += 1
            dt = time.time() - t0
            acc = correct / total
            results[name].append({
                "acc": round(acc, 3), "correct": correct, "total": total,
                "calls": calls, "calls_per_frame": round(calls / len(frames), 2),
                "wall_s": round(dt, 1),
                "s_per_frame": round(dt / len(frames), 2),
            })
            r = results[name][-1]
            print(f"  {name:<9} acc {acc:.3f} ({correct}/{total})  "
                  f"calls/frame {r['calls_per_frame']}  "
                  f"{r['s_per_frame']}s/frame")
        json.dump(results, open("adaptive_sampling_test.json", "w"), indent=2)
        print()

    llm_backend.shutdown()
    print("===== SUMMARY =====")
    print(f"{'strategy':<10} {'mean_acc':<10} {'spread':<9} "
          f"{'calls/frame':<13} {'s/frame'}")
    for name, rs in results.items():
        accs = [r["acc"] for r in rs]
        print(f"{name:<10} {st.mean(accs):<10.3f} "
              f"{max(accs)-min(accs):<9.3f} "
              f"{st.mean([r['calls_per_frame'] for r in rs]):<13.2f} "
              f"{st.mean([r['s_per_frame'] for r in rs]):.2f}")
    f3 = st.mean([r["acc"] for r in results["fixed3"]])
    ad = st.mean([r["acc"] for r in results["adaptive"]])
    cpf = st.mean([r["calls_per_frame"] for r in results["adaptive"]])
    print(f"\nadaptive vs fixed3: acc {ad:.3f} vs {f3:.3f} "
          f"(delta {ad-f3:+.3f}), calls/frame {cpf:.2f} vs 3.00")
    print("SHIP RULE: adopt adaptive if accuracy is within noise of fixed3 "
          "AND calls/frame < 3.")


if __name__ == "__main__":
    main()
