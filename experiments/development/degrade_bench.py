import json
import os
import re
import sys

import cv2
import numpy as np

FRAMES = "live_disruption_frames"
OUT = "degrade_bench.json"
CONF_FLOOR = 0.25

BASES = ["occlusion_full_baseline_020", "occlude_book_baseline_020",
         "occlude_box_baseline_020", "lighting_baseline_020",
         "identity_same_class_baseline_020"]

OCCLUDER_SRC = {"hand": "occlusion_full_disrupt_020",
                "book": "occlude_book_disrupt_020",
                "box": "occlude_box_disrupt_020",
                "jacket": "occlude_jacket_disrupt_020"}


def darken(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def blur(img, k):
    if k <= 0:
        return img
    k = int(k) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, n):
    if n <= 1:
        return img
    ker = np.zeros((n, n), np.float32)
    ker[n // 2, :] = 1.0 / n
    return cv2.filter2D(img, -1, ker)


def noise(img, sigma):
    if sigma <= 0:
        return img
    n = np.random.RandomState(0).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def downscale(img, f):
    if f >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


AXES = [
    ("darkness", [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08],
     darken, "brightness multiplier"),
    ("blur", [0, 2, 4, 7, 11, 16, 22], blur, "gaussian radius px"),
    ("motion_blur", [1, 5, 11, 21, 35, 51, 71], motion_blur, "streak px"),
    ("noise", [0, 10, 20, 35, 55, 80, 110], noise, "gaussian sigma"),
    ("downscale", [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05],
     downscale, "resolution factor"),
]

COVERAGE = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    from yolo_tracker import YoloTracker
    from run_study import make_fast_verifier
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    model = tracker.model
    verify = make_fast_verifier()

    def detect(img):
        r = model.predict(img, conf=0.05, verbose=False)[0]
        best = None
        for b in r.boxes:
            if model.names[int(b.cls)] == target:
                c = float(b.conf)
                if best is None or c > best[0]:
                    best = (c, [int(v) for v in b.xyxy[0]])
        return best

    bases = []
    for name in BASES:
        p = os.path.join(FRAMES, f"{name}.jpg")
        img = cv2.imread(p)
        if img is None:
            continue
        clean = detect(img)
        if clean is None or clean[0] < 0.5:
            print(f"  skip {name}: no confident clean detection")
            continue
        bases.append((name, img, clean))
    if not bases:
        raise SystemExit("ABORT: no usable base frames")
    print(f"{len(bases)} base frames, target={target!r}\n")

    results = {}

    for axis, levels, fn, unit in AXES:
        print(f"{axis}  ({unit})")
        print(f"  {'level':>8}{'det rate':>10}{'mean conf':>11}"
              f"{'mean IoU':>10}{'vlm yes':>9}")
        rows = []
        for lv in levels:
            confs, ious, hits, yes = [], [], 0, 0
            for name, img, clean in bases:
                d = fn(img, lv)
                got = detect(d)
                if got and got[0] >= CONF_FLOOR:
                    hits += 1
                    confs.append(got[0])
                    ious.append(iou(clean[1], got[1]))
                else:
                    confs.append(got[0] if got else 0.0)
                    ious.append(0.0)
                a = str(verify(f"Is there a {target} in this image?", d))
                yes += a.strip().lower().startswith("y")
            row = {"level": lv, "det_rate": hits / len(bases),
                   "mean_conf": round(float(np.mean(confs)), 3),
                   "mean_iou": round(float(np.mean(ious)), 3),
                   "vlm_yes": yes / len(bases)}
            rows.append(row)
            print(f"  {lv:>8}{row['det_rate']:>10.2f}{row['mean_conf']:>11.3f}"
                  f"{row['mean_iou']:>10.3f}{row['vlm_yes']:>9.2f}")
        results[axis] = {"unit": unit, "rows": rows}
        broke = next((r["level"] for r in rows if r["det_rate"] < 0.5), None)
        print(f"  -> detector breaks at {broke if broke is not None else 'never'}"
              f" ({unit})\n")

    print("coverage  (fraction of the target's height covered by a REAL "
          "occluder)")
    cov = {}
    for occ_name, src in OCCLUDER_SRC.items():
        sp = os.path.join(FRAMES, f"{src}.jpg")
        occ_img = cv2.imread(sp)
        if occ_img is None:
            continue
        base = next((b for b in bases
                     if b[0].startswith(src.split("_disrupt")[0])), None)
        if base is None:
            base = bases[0]
        name, img, clean = base
        if occ_img.shape != img.shape:
            continue
        x1, y1, x2, y2 = clean[1]
        rows = []
        print(f"  {occ_name}:")
        for c in COVERAGE:
            d = img.copy()
            if c > 0:
                cut = int(y1 + (y2 - y1) * (1 - c))
                d[cut:y2, x1:x2] = occ_img[cut:y2, x1:x2]
            got = detect(d)
            a = str(verify(f"Is there a {target} in this image?", d))
            rows.append({"coverage": c,
                         "conf": round(got[0], 3) if got else 0.0,
                         "iou": round(iou(clean[1], got[1]), 3) if got else 0.0,
                         "vlm_yes": a.strip().lower().startswith("y")})
            print(f"    {int(c * 100):>3}%  conf {rows[-1]['conf']:.3f}  "
                  f"IoU {rows[-1]['iou']:.3f}  vlm={'yes' if rows[-1]['vlm_yes'] else 'no'}")
        cov[occ_name] = rows
    results["coverage"] = cov

    json.dump(results, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
