import json
import os
import statistics
import sys
import time

import cv2

FRAMES_DIR = "live_disruption_frames"
OCC_DIR = "generated_occlusions"
N_WARM = 3


def sample_frames(n=40):
    out = []
    for fn in sorted(os.listdir(FRAMES_DIR)):
        if fn.endswith(".jpg") and "baseline" in fn:
            out.append(os.path.join(FRAMES_DIR, fn))
    return out[:n]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def best_box(model, img, target, half=False):
    kw = {"conf": 0.1, "verbose": False}
    if half:
        kw["half"] = True
    r = model.predict(img, **kw)[0]
    best = None
    for b in r.boxes:
        if model.names[int(b.cls)] == target:
            c = float(b.conf)
            if best is None or c > best[0]:
                best = (c, [int(v) for v in b.xyxy[0]])
    return best


def bench_detector(model, imgs, target, half, reps=3):
    lat, res = [], []
    for _ in range(N_WARM):
        best_box(model, imgs[0], target, half)
    for _ in range(reps):
        for img in imgs:
            t0 = time.perf_counter()
            b = best_box(model, img, target, half)
            lat.append(time.perf_counter() - t0)
            res.append(b)
    return lat, res[:len(imgs)]


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    paths = sample_frames()
    if not paths:
        raise SystemExit("ABORT: no baseline frames")
    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]

    from yolo_tracker import YoloTracker
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    model = tracker.model

    report = {}

    print("=" * 68)
    print("A. DETECTOR: fp32 vs fp16")
    print("=" * 68)
    lat32, res32 = bench_detector(model, imgs, target, half=False)
    lat16, res16 = bench_detector(model, imgs, target, half=True)
    m32, m16 = statistics.median(lat32) * 1e3, statistics.median(lat16) * 1e3

    agree = both = 0
    ious, dconf = [], []
    for a, b in zip(res32, res16):
        if (a is None) != (b is None):
            continue
        both += 1
        if a is None:
            agree += 1
            continue
        i = iou(a[1], b[1])
        ious.append(i)
        dconf.append(abs(a[0] - b[0]))
        agree += (i > 0.9)
    miss = sum((a is None) != (b is None) for a, b in zip(res32, res16))

    print(f"  fp32 median {m32:6.2f} ms")
    print(f"  fp16 median {m16:6.2f} ms   speedup {m32 / m16:.2f}x")
    print(f"  agreement: {agree}/{len(res32)} boxes at IoU>0.9, "
          f"presence flips {miss}")
    if ious:
        print(f"  median IoU {statistics.median(ious):.4f}   "
              f"max |dconf| {max(dconf):.4f}")
    report["detector"] = {"fp32_ms": round(m32, 2), "fp16_ms": round(m16, 2),
                          "speedup": round(m32 / m16, 3),
                          "iou_median": round(statistics.median(ious), 4)
                          if ious else None,
                          "presence_flips": miss,
                          "max_dconf": round(max(dconf), 4) if dconf else None}

    print("\n" + "=" * 68)
    print("B. REGION PROBE: token budget")
    print("=" * 68)
    man_path = os.path.join(OCC_DIR, "manifest.json")
    if not os.path.exists(man_path):
        print("  no generated_occlusions/manifest.json -- skipping")
    else:
        from run_study import make_fast_verifier
        from recovery_pipeline import RecoveryPipeline
        v = make_fast_verifier()
        gen = getattr(v, "generate", None)
        man = json.load(open(man_path))
        pipe = RecoveryPipeline(target, tracker=None, verifier=None,
                                use_detector=False, use_opencv=True,
                                use_vlm=False, use_state=True,
                                use_region_probe=True, probe_verifier=gen)
        pipe.box = man["box"]
        items = [(i["name"], cv2.imread(i["file"])) for i in man["items"]]
        items = [(n, im) for n, im in items if im is not None]

        rows = {}
        for tok in (12, 8, 6, 4):
            def probe_v(prompt, crop, _t=tok):
                return gen(prompt, crop, tokens=_t)
            pipe.probe_verifier = probe_v
            labs, lat = [], []
            for name, im in items:
                t0 = time.perf_counter()
                lab, is_occ = pipe.probe_region(im)
                lat.append(time.perf_counter() - t0)
                labs.append((name, lab, bool(is_occ)))
            rows[tok] = {"median_ms": round(statistics.median(lat) * 1e3, 1),
                         "labels": labs}
            print(f"  tokens={tok:<3} median {rows[tok]['median_ms']:6.1f} ms   "
                  + "  ".join(f"{n}:{l}" for n, l, _ in labs))

        base = rows[12]["labels"]
        print("\n  label agreement vs tokens=12:")
        for tok in (8, 6, 4):
            same = sum(a[2] == b[2] for a, b in zip(base, rows[tok]["labels"]))
            same_txt = sum(a[1] == b[1] for a, b in zip(base,
                                                        rows[tok]["labels"]))
            print(f"    tokens={tok:<3} classification {same}/{len(base)}   "
                  f"exact text {same_txt}/{len(base)}   "
                  f"speedup {rows[12]['median_ms'] / rows[tok]['median_ms']:.2f}x")
        report["probe"] = {str(k): {"median_ms": v["median_ms"],
                                    "labels": v["labels"]}
                           for k, v in rows.items()}

    json.dump(report, open("speed_opts.json", "w"), indent=2, default=str)
    print("\nwrote speed_opts.json")


if __name__ == "__main__":
    main()
