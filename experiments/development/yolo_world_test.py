import json
import time

import cv2

from heldout_gt import HELDOUT_GT, HALLUCINATION_PROBES

QUERIES = ["laptop", "keyboard", "computer mouse", "water bottle"]
PROBES = list(HALLUCINATION_PROBES)
CLASSES = QUERIES + PROBES

MODELS = ["yolov8s-worldv2.pt", "yolov8x-worldv2.pt"]

CONF_SWEEP = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
RAW = "yolo_world_raw.json"


def collect(model_name):
    from ultralytics import YOLOWorld
    model = YOLOWorld(model_name)
    model.set_classes(CLASSES)

    out, lat = {}, []
    for frame in sorted(HELDOUT_GT):
        img = cv2.imread(frame)
        if img is None:
            print(f"  {frame}: unreadable, skipping")
            continue
        t0 = time.time()
        res = model.predict(img, conf=0.001, verbose=False)[0]
        dt = time.time() - t0
        lat.append(dt)

        dets = []
        for b in res.boxes:
            dets.append({
                "cls": model.names[int(b.cls.item())],
                "conf": round(float(b.conf.item()), 4),
                "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
            })
        dets.sort(key=lambda d: -d["conf"])
        out[frame] = {"dets": dets, "latency_s": round(dt, 3)}
        top = ", ".join(f"{d['cls']}:{d['conf']:.2f}" for d in dets[:6])
        print(f"  {frame:<26} {dt:5.2f}s  {len(dets):3d} dets | {top}")

    med = sorted(lat)[len(lat) // 2] if lat else None
    print(f"  median latency {med:.3f}s over {len(lat)} frames")
    return out, med


def score_at(data, conf):
    tp = fp = tn = fn = 0
    for frame, labels in HELDOUT_GT.items():
        if frame not in data:
            continue
        fired = {d["cls"] for d in data[frame]["dets"] if d["conf"] >= conf}
        for obj, gt in labels.items():
            pred = obj in fired
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1
    p, n = tp + fn, tn + fp
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": round((tp + tn) / (p + n), 3) if (p + n) else None,
        "sens": round(tp / p, 3) if p else None,
        "spec": round(tn / n, 3) if n else None,
    }


def probe_hits(data, conf):
    hits = []
    for frame in data:
        for d in data[frame]["dets"]:
            if d["conf"] >= conf and d["cls"] in PROBES:
                hits.append(f"{frame}: {d['cls']} @ {d['conf']:.3f}")
    return hits


def report(model_name, data, med):
    print("\n" + "=" * 78)
    print(f"{model_name} -- HELD-OUT (16 points: 11 present / 5 absent, 4 frames)")
    print("=" * 78)
    hdr = f"{'conf':>7}{'acc':>7}{'sens':>7}{'spec':>7}{'tp':>4}{'fp':>4}{'tn':>4}{'fn':>4}  probes"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for c in CONF_SWEEP:
        m = score_at(data, c)
        rows[str(c)] = m
        print(f"{c:>7}{m['acc']:>7}{m['sens']:>7}{m['spec']:>7}"
              f"{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}"
              f"  {len(probe_hits(data, c))}")

    print("\nVLM reference (3-run means, same 16 points):")
    print("  whole_maj3 (SHIPPED)   acc 0.750   ~3.2 s/frame")
    print("  crops_union (opt-in)   acc 0.875   ~6 calls/frame")
    print("  wholemaj3+cropsvote2   acc 0.896   ~8 calls/frame")

    print("\nwater-bottle trace (the documented VLM weak spot):")
    for frame in sorted(data):
        gt = HELDOUT_GT[frame]["water bottle"]
        best = max((d["conf"] for d in data[frame]["dets"]
                    if d["cls"] == "water bottle"), default=0.0)
        print(f"  {frame:<26} gt={gt}  best_conf={best:.3f}")

    print("\ncomputer-mouse trace (systematic headphones->mouse FP on c2_09):")
    for frame in sorted(data):
        gt = HELDOUT_GT[frame]["computer mouse"]
        best = max((d["conf"] for d in data[frame]["dets"]
                    if d["cls"] == "computer mouse"), default=0.0)
        print(f"  {frame:<26} gt={gt}  best_conf={best:.3f}")

    ph = probe_hits(data, 0.05)
    print(f"\nhallucination probes @0.05: {len(ph)}")
    for h in ph[:10]:
        print(f"  {h}")
    return rows


if __name__ == "__main__":
    all_out = {}
    for mn in MODELS:
        print(f"\n### {mn}")
        try:
            data, med = collect(mn)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        rows = report(mn, data, med)
        all_out[mn] = {"raw": data, "scores": rows, "median_latency_s": med}
        json.dump(all_out, open(RAW, "w"), indent=2)
    print(f"\nwrote {RAW}")
