import json
import os
import time

import cv2

import llm_backend
import open_vocab_detect as ov
from region_proposer import propose_regions
from heldout_gt import HELDOUT_GT, HALLUCINATION_PROBES

RESULTS = "proposer_verify_results.json"
WHOLE_SAMPLES = 3

BLIND = {
    "TL":     (0.00, 0.60, 0.00, 0.60),
    "TR":     (0.00, 0.60, 0.40, 1.00),
    "BL":     (0.40, 1.00, 0.00, 0.60),
    "BR":     (0.40, 1.00, 0.40, 1.00),
    "center": (0.20, 0.80, 0.25, 0.75),
}


def crop(img, box):
    h, w = img.shape[:2]
    r0, r1, c0, c1 = box
    return img[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def list_once(img_bgr):
    t0 = time.time()
    txt = llm_backend.vision_chat(ov.LIST_PROMPT, img_bgr, max_tokens=ov.MAX_TOKENS)
    dt = time.time() - t0
    items = ov._dedupe(ov._parse_items(txt)) if txt else []
    return items, round(dt, 2)


def collect():
    if not llm_backend.ensure_running():
        raise SystemExit("llama-server not healthy; aborting")
    data = {}
    for frame in sorted(HELDOUT_GT):
        img = cv2.imread(frame)
        if img is None:
            print(f"  {frame}: unreadable, skipping")
            continue
        print(f"\n=== {frame}  ({img.shape[1]}x{img.shape[0]}) ===")
        rec = {"whole": [], "blind": {}, "prop": {}, "prop_boxes": []}
        for i in range(WHOLE_SAMPLES):
            items, dt = list_once(img)
            rec["whole"].append(items)
            print(f"  whole[{i}] ({dt}s): {items}")
        for name, box in BLIND.items():
            items, dt = list_once(crop(img, box))
            rec["blind"][name] = items
            print(f"  blind {name:<7} ({dt}s): {items}")
        regions = propose_regions(img)
        rec["prop_boxes"] = regions
        for i, box in enumerate(regions):
            items, dt = list_once(crop(img, box))
            rec["prop"][f"r{i}"] = items
            print(f"  prop  r{i} {box} ({dt}s): {items}")
        data[frame] = rec
        json.dump(data, open(RESULTS, "w"), indent=2)
    return data


def present_any(query, lists):
    return any(ov.is_present(query, l) for l in lists)


def votes(query, lists):
    return sum(1 for l in lists if ov.is_present(query, l))


def score(data):
    conds = {
        "whole_maj3(base)":  lambda q, r: votes(q, r["whole"]) >= 2,
        "blind_union":       lambda q, r: present_any(q, list(r["blind"].values())),
        "blind_vote2":       lambda q, r: votes(q, list(r["blind"].values())) >= 2,
        "prop_union":        lambda q, r: present_any(q, list(r["prop"].values())),
        "prop_vote2":        lambda q, r: votes(q, list(r["prop"].values())) >= 2,
        "wholemaj3+prop_v2": lambda q, r: votes(q, r["whole"]) >= 2
                                          or votes(q, list(r["prop"].values())) >= 2,
        "wholemaj3+blind_v2": lambda q, r: votes(q, r["whole"]) >= 2
                                          or votes(q, list(r["blind"].values())) >= 2,
    }
    rows = {}
    for cname, fn in conds.items():
        tp = fp = tn = fn_ = 0
        for frame, labels in HELDOUT_GT.items():
            if frame not in data:
                continue
            rec = data[frame]
            for obj, gt in labels.items():
                pred = fn(obj, rec)
                if gt and pred:
                    tp += 1
                elif gt and not pred:
                    fn_ += 1
                elif not gt and pred:
                    fp += 1
                else:
                    tn += 1
        p, n = tp + fn_, tn + fp
        rows[cname] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn_,
                       "acc": round((tp + tn) / (p + n), 3) if (p + n) else None,
                       "sens": round(tp / p, 3) if p else None,
                       "spec": round(tn / n, 3) if n else None}

    halluc = {"blind": 0, "prop": 0, "detail": []}
    for frame in data:
        bl = list(data[frame]["blind"].values())
        pl = list(data[frame]["prop"].values())
        for probe in HALLUCINATION_PROBES:
            if present_any(probe, bl):
                halluc["blind"] += 1
                halluc["detail"].append(f"{frame} blind: {probe}")
            if present_any(probe, pl):
                halluc["prop"] += 1
                halluc["detail"].append(f"{frame} prop: {probe}")

    cost = {}
    for frame in data:
        cost[frame] = {"blind_crops": len(data[frame]["blind"]),
                       "prop_crops": len(data[frame]["prop"])}
    return rows, halluc, cost


def report(data):
    rows, halluc, cost = score(data)
    print("\n" + "=" * 78)
    print("HELD-OUT SCORING (16 points: 11 present / 5 absent, 4 c2 frames)")
    print("=" * 78)
    hdr = f"{'condition':<22}{'acc':>6}{'sens':>7}{'spec':>7}{'tp':>4}{'fp':>4}{'tn':>4}{'fn':>4}"
    print(hdr); print("-" * len(hdr))
    for cname, m in rows.items():
        print(f"{cname:<22}{m['acc']:>6}{m['sens']:>7}{m['spec']:>7}"
              f"{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}")
    print(f"\nhallucination mentions  blind={halluc['blind']}  prop={halluc['prop']}")
    for d in halluc["detail"]:
        print(f"  {d}")

    print("\ncrop-call COST per frame (whole adds +3 to any whole-using cond):")
    tb = tp_ = 0
    for frame in sorted(cost):
        c = cost[frame]
        tb += c["blind_crops"]; tp_ += c["prop_crops"]
        print(f"  {frame:<26} blind={c['blind_crops']}  prop={c['prop_crops']}")
    print(f"  TOTAL crop calls          blind={tb}  prop={tp_}"
          f"  ({100*tp_/tb:.0f}% of blind)")

    print("\nwater-bottle trace (the FN crop-and-verify exists to recover):")
    for frame in sorted(data):
        rec = data[frame]
        gt = HELDOUT_GT[frame]["water bottle"]
        wv = votes("water bottle", rec["whole"])
        bh = [k for k, v in rec["blind"].items() if ov.is_present("water bottle", v)]
        ph = [k for k, v in rec["prop"].items() if ov.is_present("water bottle", v)]
        print(f"  {frame:<26} gt={gt}  whole={wv}/3  blind_hit={bh}  prop_hit={ph}")

    print("\ncomputer-mouse trace (systematic headphones->mouse FP on c2_09):")
    for frame in sorted(data):
        rec = data[frame]
        gt = HELDOUT_GT[frame]["computer mouse"]
        bh = [k for k, v in rec["blind"].items() if ov.is_present("computer mouse", v)]
        ph = [k for k, v in rec["prop"].items() if ov.is_present("computer mouse", v)]
        print(f"  {frame:<26} gt={gt}  blind_hit={bh}  prop_hit={ph}")

    json.dump({"scores": rows, "halluc": halluc, "cost": cost},
              open("proposer_verify_scores.json", "w"), indent=2)


if __name__ == "__main__":
    if os.path.exists(RESULTS):
        print(f"[re-scoring existing {RESULTS}; delete to re-collect]")
        data = json.load(open(RESULTS))
    else:
        data = collect()
    report(data)
