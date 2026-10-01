import glob
import json
import statistics as st
import time

import cv2

import llm_backend
from open_listing_test import SYNONYMS, in_list, parse_items
from openvocab_baseline import GT, VOCAB

MAX_ITEMS = 25

VARIANTS = {
    "current": ("List every distinct physical object you can clearly see in "
                "this image. Reply with ONLY a comma-separated list of short "
                "object names, nothing else."),
    "exhaustive": ("List ALL physical objects visible in this image, "
                   "including small ones, partially visible ones, and objects "
                   "in the background or at the edges. Be thorough and do not "
                   "stop early. Reply with ONLY a comma-separated list of "
                   "short object names."),
    "count_hint": ("Carefully scan this whole image and list at least 8 "
                   "different physical objects you can see, from the largest "
                   "to the smallest. Include partially visible objects. Reply "
                   "with ONLY a comma-separated list of short object names."),
    "regions": ("Scan this image from left to right. List the physical "
                "objects on the left side, then the middle, then the right "
                "side, including small and partially visible ones. Reply with "
                "ONLY one comma-separated list of short object names."),
}


def dedupe(items):
    seen, out = set(), []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= MAX_ITEMS:
            break
    return out


def score(rows):
    bal, per = [], {}
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
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        if sens is not None and spec is not None:
            bal.append((sens + spec) / 2)
            per[obj] = {"bal": round((sens + spec) / 2, 3),
                        "sens": round(sens, 3), "spec": round(spec, 3)}
        else:
            per[obj] = {"bal": None, "sens": sens, "spec": spec}
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, per)


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    print(f"LISTING RECALL: {len(VARIANTS)} prompts x {len(frames)} frames "
          f"(1 call each)\n")

    summary = {}
    for name, prompt in VARIANTS.items():
        print(f"##### {name} #####")
        rows, lat, counts = [], [], []
        for f in frames:
            idx = idx_of[f]
            t0 = time.time()
            out = llm_backend.vision_chat(prompt, imgs[f], max_tokens=140)
            dt = time.time() - t0
            lat.append(dt)
            items = dedupe(parse_items(out))
            counts.append(len(items))
            for obj in VOCAB:
                rows.append({"idx": idx, "obj": obj, "gt": GT[idx][obj],
                             "pred": 1 if in_list(obj, items) else 0})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {dt:5.1f}s {got}/{len(VOCAB)} "
                  f"({len(items)} items) {items[:8]}")
        macro, fp, per = score(rows)
        summary[name] = {
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp, 3) if fp is not None else None,
            "mean_items": round(st.mean(counts), 1),
            "med_lat_s": round(st.median(lat), 2),
            "per_object": per,
        }
        json.dump(summary, open("listing_recall_test.json", "w"), indent=2)
        s = summary[name]
        print(f"  -> macro {s['macro_bal_acc']}  FP {s['fp_rate']}  "
              f"mean_items {s['mean_items']}  med {s['med_lat_s']}s\n")

    llm_backend.shutdown()
    print("===== LISTING RECALL SUMMARY =====")
    print(f"{'variant':<12} {'macro':<8} {'fp_rate':<9} {'mean_items':<11} "
          f"{'med_s'}")
    for n, s in summary.items():
        print(f"{n:<12} {str(s['macro_bal_acc']):<8} {str(s['fp_rate']):<9} "
              f"{str(s['mean_items']):<11} {s['med_lat_s']}")
    print("\nTargets: beat macro 0.785 / FP 0.114 (current listing prompt).")
    print("Per-object sensitivity (recall) is the number to watch:")
    for n, s in summary.items():
        sens = {o: v["sens"] for o, v in s["per_object"].items()}
        print(f"  {n:<12} {sens}")


if __name__ == "__main__":
    main()
