import glob
import json
import re
import statistics as st
import time

import cv2

import llm_backend
from openvocab_baseline import GT, VOCAB

PROMPT = ("List every distinct physical object you can clearly see in this "
          "image. Reply with ONLY a comma-separated list of short object "
          "names, nothing else.")

SYNONYMS = {
    "water bottle": ["water bottle", "bottle", "flask", "tumbler",
                     "thermos", "canteen"],
    "keyboard": ["keyboard", "keypad"],
    "computer mouse": ["computer mouse", "mouse"],
    "laptop": ["laptop", "notebook computer", "macbook"],
    "banana": ["banana"],
}
BLOCK_SUBSTR = {
    "computer mouse": ["mousepad", "mouse pad", "mouse mat"],
    "keyboard": ["keyboard tray"],
}


def in_list(obj, items):
    for raw in items:
        s = raw.strip().lower()
        if not s:
            continue
        if any(b in s for b in BLOCK_SUBSTR.get(obj, [])):
            continue
        for syn in SYNONYMS[obj]:
            if re.search(rf"\b{re.escape(syn)}\b", s):
                return True
    return False


def parse_items(text):
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^[^:]{0,40}:", "", t)
    parts = re.split(r"[,\n;]+", t)
    out = []
    for p in parts:
        p = re.sub(r"^\s*[-*\d.)]+\s*", "", p).strip()
        if p:
            out.append(p)
    return out


def balanced_accuracy(rows):
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
            per[obj] = round((sens + spec) / 2, 3)
        else:
            per[obj] = None
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, per)


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    print(f"OPEN LISTING: 1 call per frame ({len(frames)} calls) vs "
          f"{len(frames)*len(VOCAB)} for per-object\n")

    rows, lat, raws = [], [], {}
    for f in frames:
        idx = idx_of[f]
        t0 = time.time()
        out = llm_backend.vision_chat(PROMPT, imgs[f], max_tokens=100)
        dt = time.time() - t0
        lat.append(dt)
        items = parse_items(out)
        raws[f] = {"raw": out, "items": items, "lat_s": round(dt, 2)}
        line = []
        for obj in VOCAB:
            pred = 1 if in_list(obj, items) else 0
            gt = GT[idx][obj]
            rows.append({"idx": idx, "obj": obj, "gt": gt, "pred": pred})
            line.append(f"{obj.split()[-1]}:{'OK' if pred == gt else 'XX'}")
        print(f"  [{idx:02d}] {dt:5.1f}s {' '.join(line)}")
        print(f"       listed: {items}")

    llm_backend.shutdown()
    macro, fp_rate, per = balanced_accuracy(rows)
    res = {
        "macro_bal_acc": round(macro, 3) if macro else None,
        "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "per_object": per,
        "calls": len(frames),
        "med_lat_per_frame_s": round(st.median(lat), 2),
        "total_lat_s": round(sum(lat), 1),
        "raw": raws,
    }
    json.dump(res, open("open_listing_test.json", "w"), indent=2)

    print("\n===== OPEN LISTING RESULT =====")
    print(f"macro balanced acc : {res['macro_bal_acc']}   (baseline 0.753)")
    print(f"FP rate            : {res['fp_rate']}   (baseline 0.371)")
    print(f"per-object         : {per}")
    print(f"calls              : {len(frames)}  (per-object needs "
          f"{len(frames)*len(VOCAB)})")
    print(f"latency/frame      : median {res['med_lat_per_frame_s']}s, "
          f"total {res['total_lat_s']}s")
    print("\nNOTE: per-frame latency covers ALL 5 objects here, so compare "
          "against 5x the per-object figure.")


if __name__ == "__main__":
    main()
