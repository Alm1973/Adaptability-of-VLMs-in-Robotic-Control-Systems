import json
import os
import statistics
import sys
import time

import cv2

FRAMES_DIR = "live_disruption_frames"
OCC_DIR = "generated_occlusions"
REPS = 5


def sample_frames(n=30):
    out = [os.path.join(FRAMES_DIR, f) for f in sorted(os.listdir(FRAMES_DIR))
           if f.endswith(".jpg") and "baseline" in f]
    return out[:n]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = (max(ax1, bx1), max(ay1, by1),
                          min(ax2, bx2), min(ay2, by2))
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def detect(model, img, target, **kw):
    r = model.predict(img, conf=0.1, verbose=False, **kw)[0]
    best = None
    for b in r.boxes:
        if model.names[int(b.cls)] == target:
            c = float(b.conf)
            if best is None or c > best[0]:
                best = (c, [int(v) for v in b.xyxy[0]])
    return best


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    imgs = [cv2.imread(p) for p in sample_frames()]
    imgs = [i for i in imgs if i is not None]

    from yolo_tracker import YoloTracker
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    model = tracker.model
    report = {}

    print("=" * 70)
    print("C. DETECTOR INPUT SIZE (real compute reduction)")
    print("=" * 70)
    configs = [("imgsz=640 (default)", {}),
               ("imgsz=576", {"imgsz": 576}),
               ("imgsz=512", {"imgsz": 512}),
               ("imgsz=448", {"imgsz": 448})]
    for _ in range(3):
        detect(model, imgs[0], target)

    lat = {name: [] for name, _ in configs}
    out = {name: None for name, _ in configs}
    for rep in range(REPS):
        for name, kw in configs:
            res = []
            for img in imgs:
                t0 = time.perf_counter()
                res.append(detect(model, img, target, **kw))
                lat[name].append(time.perf_counter() - t0)
            if rep == 0:
                out[name] = res

    base = out["imgsz=640 (default)"]
    base_ms = statistics.median(lat["imgsz=640 (default)"]) * 1e3
    print(f"{'config':<22}{'median ms':>11}{'speedup':>9}"
          f"{'IoU>0.9':>10}{'flips':>7}{'medIoU':>9}")
    print("-" * 70)
    for name, _ in configs:
        ms = statistics.median(lat[name]) * 1e3
        res = out[name]
        flips = sum((a is None) != (b is None) for a, b in zip(base, res))
        ious = [iou(a[1], b[1]) for a, b in zip(base, res)
                if a is not None and b is not None]
        ok = sum(i > 0.9 for i in ious)
        med = statistics.median(ious) if ious else 0.0
        print(f"{name:<22}{ms:>11.2f}{base_ms / ms:>9.2f}x"
              f"{ok:>7}/{len(ious):<3}{flips:>7}{med:>9.4f}")
        report.setdefault("detector_imgsz", {})[name] = {
            "median_ms": round(ms, 2), "speedup": round(base_ms / ms, 3),
            "iou_gt09": ok, "n": len(ious), "presence_flips": flips,
            "median_iou": round(med, 4)}

    man_path = os.path.join(OCC_DIR, "manifest.json")
    if os.path.exists(man_path):
        print("\n" + "=" * 70)
        print("H2. IS THE PROBE'S TOKEN CAP EVER REACHED?")
        print("=" * 70)
        from run_study import make_fast_verifier
        v = make_fast_verifier()
        gen = getattr(v, "generate", None)
        man = json.load(open(man_path))
        x1, y1, x2, y2 = man["box"]
        pad = 20
        crops = []
        for it in man["items"]:
            im = cv2.imread(it["file"])
            if im is None:
                continue
            H, W = im.shape[:2]
            crops.append((it["name"], im[max(0, y1 - pad):min(H, y2 + pad),
                                         max(0, x1 - pad):min(W, x2 + pad)]))
        for name, crop in crops:
            ans = gen("What is the main object in this image? Answer with one "
                      "or two words only.", crop, tokens=12)
            words = len(str(ans).split())
            print(f"  {name:<8} -> {str(ans)[:24]!r:<26} ~{words} word(s)")
        print("\n  If every answer is 1-2 words, the model hits EOS long before")
        print("  12 tokens, so lowering max_new_tokens cannot save time it")
        print("  never spent -- which is why that series was non-monotonic.")

    json.dump(report, open("speed_opts2.json", "w"), indent=2, default=str)
    print("\nwrote speed_opts2.json")


if __name__ == "__main__":
    main()
