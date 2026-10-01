import json
import os
import sys

import cv2
import numpy as np

FRAMES = "live_disruption_frames"
OUT = "search_challenge.json"
CONF_FLOOR = 0.25

BASES = ["occlusion_full_baseline_020", "occlude_book_baseline_020",
         "occlude_box_baseline_020", "identity_same_class_baseline_020",
         "lighting_baseline_020"]

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]


def clip_edge(img, box, f, side="right"):
    if f <= 0:
        return img
    x1, y1, x2, y2 = box
    w = x2 - x1
    h, W = img.shape[:2]
    if side == "right":
        shift = int(x1 + f * w)
        shift = max(0, min(W - 1, shift))
        if shift == 0:
            return img
        return np.hstack([img[:, shift:],
                          np.zeros((h, shift, 3), np.uint8)])
    shift = int((W - x2) + f * w)
    shift = max(0, min(W - 1, shift))
    if shift == 0:
        return img
    return np.hstack([np.zeros((h, shift, 3), np.uint8), img[:, :W - shift]])


def cover(img, box, f, direction):
    if f <= 0:
        return img
    x1, y1, x2, y2 = box
    out = img.copy()
    src = cv2.imread(os.path.join(FRAMES, "occlude_box_disrupt_020.jpg"))
    if src is None or src.shape != img.shape:
        return out
    if direction == "top":
        cut = int(y1 + (y2 - y1) * (1 - f))
        out[cut:y2, x1:x2] = src[cut:y2, x1:x2]
    elif direction == "bottom":
        cut = int(y1 + (y2 - y1) * f)
        out[y1:cut, x1:x2] = src[y1:cut, x1:x2]
    else:
        cut = int(x1 + (x2 - x1) * f)
        out[y1:y2, x1:cut] = src[y1:y2, x1:cut]
    return out


def same_colour(img, box, f):
    if f <= 0:
        return img
    x1, y1, x2, y2 = box
    out = img.copy()
    patch = cv2.flip(img[y1:y2, x1:x2], 1)
    cut = int(y1 + (y2 - y1) * (1 - f))
    ph = cut - y1
    if ph < 0 or cut >= y2:
        return out
    out[cut:y2, x1:x2] = patch[ph:, :]
    return out


def local_shadow(img, box, f):
    if f <= 0:
        return img
    x1, y1, x2, y2 = box
    out = img.copy().astype(np.float32)
    pad = 18
    m = np.zeros(img.shape[:2], np.float32)
    m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 1.0
    m = cv2.GaussianBlur(m, (41, 41), 0)
    factor = 1.0 - f * 0.95
    out *= (1.0 - m[..., None] * (1.0 - factor))
    return np.clip(out, 0, 255).astype(np.uint8)


def glare(img, box, f):
    if f <= 0:
        return img
    x1, y1, x2, y2 = box
    out = img.copy().astype(np.float32)
    m = np.zeros(img.shape[:2], np.float32)
    cx, cy = (x1 + x2) // 2, y1 + int((y2 - y1) * 0.35)
    r = int(max(x2 - x1, y2 - y1) * 0.55)
    cv2.circle(m, (cx, cy), max(4, r), 1.0, -1)
    m = cv2.GaussianBlur(m, (61, 61), 0)
    out += m[..., None] * 255.0 * f
    return np.clip(out, 0, 255).astype(np.uint8)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
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
    for n in BASES:
        img = cv2.imread(os.path.join(FRAMES, f"{n}.jpg"))
        if img is None:
            continue
        d = detect(img)
        if d and d[0] >= 0.5:
            bases.append((n, img, d))
    if not bases:
        raise SystemExit("ABORT: no usable base frames")
    print(f"{len(bases)} base frames, target={target!r}\n")

    CHALLENGES = [
        ("edge_clip_left", lambda i, b, f: clip_edge(i, b, f, "right"),
         "target slides off the LEFT frame edge (camera pans right)"),
        ("edge_clip_right", lambda i, b, f: clip_edge(i, b, f, "left"),
         "target slides off the RIGHT frame edge (camera pans left)"),
        ("cover_top", lambda i, b, f: cover(i, b, f, "top"),
         "rim visible, body hidden"),
        ("cover_bottom", lambda i, b, f: cover(i, b, f, "bottom"),
         "RIM HIDDEN, body visible"),
        ("cover_side", lambda i, b, f: cover(i, b, f, "side"),
         "half the silhouette hidden"),
        ("local_shadow", local_shadow, "shadow across the target only"),
        ("glare", glare, "specular blowout on the target"),
    ]

    results = {}
    print(f"{'challenge':<18}" + "".join(f"{int(l * 100):>6}%" for l in LEVELS)
          + "   breaks at")
    print("-" * 74)
    for name, fn, desc in CHALLENGES:
        rates, rows = [], []
        for lv in LEVELS:
            hits, confs, ious, yes = 0, [], [], 0
            for bn, img, clean in bases:
                d = fn(img, clean[1], lv)
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
            rate = hits / len(bases)
            rates.append(rate)
            rows.append({"level": lv, "det_rate": rate,
                         "mean_conf": round(float(np.mean(confs)), 3),
                         "mean_iou": round(float(np.mean(ious)), 3),
                         "vlm_yes": yes / len(bases)})
        broke = next((LEVELS[i] for i, r in enumerate(rates) if r < 0.5), None)
        results[name] = {"desc": desc, "rows": rows,
                         "breaks_at": broke}
        print(f"{name:<18}" + "".join(f"{r:>6.2f}" for r in rates)
              + f"   {'never' if broke is None else f'{int(broke * 100)}%'}")

    print("\n" + "=" * 74)
    print("RANKED BY DIFFICULTY (earliest break = hardest)")
    print("=" * 74)
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1]["breaks_at"] is None,
                                    kv[1]["breaks_at"] or 1.0))
    for name, r in ranked:
        b = r["breaks_at"]
        print(f"  {name:<18}{('breaks at ' + str(int(b * 100)) + '%') if b is not None else 'never breaks':<20}"
              f"{r['desc']}")
    print("\n  known limits for comparison: apparent size breaks at 0.12x "
          "resolution,\n  top-down coverage breaks at 85%.")

    json.dump(results, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
