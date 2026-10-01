import glob
import json
import re
import statistics as st
import time
from collections import Counter

import cv2

import llm_backend
from open_vocab_detect import LIST_PROMPT, _dedupe, _parse_items, is_present
from openvocab_baseline import GT, VOCAB

N_SAMPLES = 3
REPEATS = 2

HALLUC = {"banana": ["banana"], "bicycle": ["bicycle", "bike"],
          "elephant": ["elephant"], "umbrella": ["umbrella"]}
VERIFIED = {
    "c2_07_tilt_b90_t78.jpg": {"laptop": 1, "water bottle": 1,
                               "keyboard": 0, "computer mouse": 0},
    "c2_10_home.jpg":         {"laptop": 1, "water bottle": 1,
                               "keyboard": 1, "computer mouse": 1},
}


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


def probe_hit_lists(p, lists, mode):
    hits = 0
    for l in lists:
        if any(re.search(rf"\b{re.escape(s)}\b", it.strip().lower())
               for it in l for s in HALLUC[p]):
            hits += 1
    return hits >= (2 if mode == "major3" else 1)


def score_insample(rows):
    bal, sens = [], {}
    all_neg = all_fp = 0
    for obj in VOCAB:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos, nneg = tp + fn, tn + fp
        all_neg += nneg
        all_fp += fp
        s = tp / npos if npos else None
        sp = tn / nneg if nneg else None
        sens[obj] = round(s, 3) if s is not None else None
        if s is not None and sp is not None:
            bal.append((s + sp) / 2)
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, sens)


def main():
    exp = sorted(glob.glob("exp_*.jpg"))
    c2 = sorted(glob.glob("c2_*.jpg"))
    imgs = {f: cv2.imread(f) for f in exp + c2}
    idx_of = {f: i + 1 for i, f in enumerate(exp)}
    print(f"TEMPORAL VOTE: {N_SAMPLES} samples/frame, {REPEATS} repeats "
          f"for stability\n")

    results = {}
    for rep in range(1, REPEATS + 1):
        print(f"===== repeat {rep}/{REPEATS} =====")
        t0 = time.time()
        exp_lists = {f: sample_lists(imgs[f], N_SAMPLES) for f in exp}
        c2_lists = {f: sample_lists(imgs[f], N_SAMPLES) for f in c2}
        elapsed = time.time() - t0

        for mode in ("single", "union3", "major3"):
            rows = []
            for f in exp:
                idx = idx_of[f]
                ls = exp_lists[f][:1] if mode == "single" else exp_lists[f]
                for obj in VOCAB:
                    rows.append({"obj": obj, "gt": GT[idx][obj],
                                 "pred": decide(obj, ls, mode)})
            macro, fp, sens = score_insample(rows)

            hits = checks = 0
            for f in c2:
                ls = c2_lists[f][:1] if mode == "single" else c2_lists[f]
                for p in HALLUC:
                    checks += 1
                    hits += probe_hit_lists(p, ls, mode)
            correct = total = 0
            for f, labels in VERIFIED.items():
                ls = c2_lists[f][:1] if mode == "single" else c2_lists[f]
                for obj, gt in labels.items():
                    total += 1
                    if decide(obj, ls, mode) == gt:
                        correct += 1

            results.setdefault(mode, []).append({
                "repeat": rep,
                "in_macro": round(macro, 3) if macro else None,
                "in_fp": round(fp, 3) if fp is not None else None,
                "bottle_sens": sens.get("water bottle"),
                "held_halluc": f"{hits}/{checks}",
                "held_positives": f"{correct}/{total}",
            })
            r = results[mode][-1]
            print(f"  {mode:<8} macro {r['in_macro']} FP {r['in_fp']} "
                  f"bottle {r['bottle_sens']} | HELD pos "
                  f"{r['held_positives']} hal {r['held_halluc']}")
        print(f"  (sampling for this repeat took {elapsed:.0f}s)\n")
        json.dump(results, open("temporal_vote_test.json", "w"), indent=2)

    llm_backend.shutdown()
    print("===== STABILITY ACROSS REPEATS (the real question) =====")
    print(f"{'mode':<8} {'macro runs':<20} {'spread':<9} "
          f"{'bottle runs':<16} {'HELD pos'}")
    for mode, runs in results.items():
        macros = [r["in_macro"] for r in runs if r["in_macro"] is not None]
        bottles = [r["bottle_sens"] for r in runs
                   if r["bottle_sens"] is not None]
        spread = round(max(macros) - min(macros), 3) if len(macros) > 1 else None
        print(f"{mode:<8} {str(macros):<20} {str(spread):<9} "
              f"{str(bottles):<16} {[r['held_positives'] for r in runs]}")
    print("\nBaseline instability to beat: the SAME prompt gave bottle "
          "sensitivity 0.444 / 0.333 / 0.000 on three separate runs.")
    print("SHIP RULE: held-out positives > 6/8 AND held-out hallucinations "
          "0/40. Lower macro spread is the secondary win.")


if __name__ == "__main__":
    main()
