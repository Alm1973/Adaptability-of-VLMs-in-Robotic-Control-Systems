import json
import os
import random

import cv2
import numpy as np

BASELINE_FRAMES = 6
RECOVER_FRAMES = 8

DISRUPT_LENGTHS = (4, 8, 14)

REPEATS = 3
SEED = 0
OUT_DIR = "episodes"

DISTRACTOR_CLASSES = ["carton", "soda can", "water bottle", "mug",
                      "computer mouse", "headphones", "keyboard"]
DISTRACTOR_MIN_CONF = 0.12

IMPOSTOR_HUE_SHIFTS = (90, 75, 60, 45, 30)
IMPOSTOR_MIN_CONF = 0.20

TARGET_NAME = ["unknown"]


def _target_box(frame, target, tracker=None):
    from yolo_tracker import YoloTracker
    t = tracker or YoloTracker(default_target=target)
    t.set_targets([target], allow_unreliable=True, quiet=True)
    res = t.model.predict(frame, conf=0.10, verbose=False)[0]
    best = None
    for b in res.boxes:
        if t.model.names[int(b.cls.item())] != target:
            continue
        c = float(b.conf.item())
        if best is None or c > best[0]:
            best = (c, [int(v) for v in b.xyxy[0].tolist()])
    return best


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


DISTRACTOR_MAX_IOU = 0.30


def _clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


OCCLUDER_COLOURS = [(28, 28, 30), (52, 48, 45), (95, 92, 88),
                    (140, 138, 134), (186, 184, 180)]


def _occlude(frame, box, fraction, colour=(28, 28, 30), rng=None):
    out = frame.copy()
    if fraction <= 0:
        return out
    x1, y1, x2, y2 = box
    pad = int((x2 - x1) * 0.12)
    cover = int((y2 - y1) * min(1.0, fraction))
    px1, py1 = x1 - pad, y2 - cover
    px2, py2 = x2 + pad, y2 + pad
    cv2.rectangle(out, (px1, py1), (px2, py2), colour, -1)

    cx1, cy1, cx2, cy2 = _clip_box((px1, py1, px2, py2), out.shape)
    if cx2 > cx1 and cy2 > cy1:
        patch = out[cy1:cy2, cx1:cx2].astype(np.float32)
        r = rng or random
        noise = np.random.default_rng(r.randrange(1 << 30)).normal(
            0, 9.0, patch.shape)
        out[cy1:cy2, cx1:cx2] = np.clip(patch + noise, 0, 255).astype(np.uint8)
    return out


def _remove_target(frame, box):
    x1, y1, x2, y2 = box
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.rectangle(mask, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 255, -1)
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _swap_instance(frame, box, hue_shift):
    out = frame.copy()
    x1, y1, x2, y2 = _clip_box(box, frame.shape)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return out
    roi = out[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    body = (s > 70) & (v > 45)
    if body.sum() < 40:
        return out
    h[body] = ((h[body].astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    out[y1:y2, x1:x2] = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return out


def build_impostors(frame, box, target):
    out = []
    for shift in IMPOSTOR_HUE_SHIFTS:
        img = _swap_instance(frame, box, shift)
        got = _target_box(img, target)
        if got and got[0] >= IMPOSTOR_MIN_CONF:
            out.append((img, shift, got[0]))
    return out


def _relight(frame, factor):
    return cv2.convertScaleAbs(frame, alpha=factor, beta=(factor - 1.0) * 40)


def _white_balance(frame, k):
    b, g, r = cv2.split(frame.astype(np.float32))
    r *= (1.0 - k)
    b *= (1.0 + k)
    g *= (1.0 - 0.15 * k)
    return cv2.merge([np.clip(c, 0, 255) for c in (b, g, r)]).astype(np.uint8)


def _shadow(frame, strength, origin):
    h, w = frame.shape[:2]
    ox, oy = origin
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - ox) / w) ** 2 + ((yy - oy) / h) ** 2)
    mask = np.clip(1.0 - strength * np.clip(1.0 - d / 0.75, 0, 1), 0, 1)
    return np.clip(frame.astype(np.float32) * mask[..., None],
                   0, 255).astype(np.uint8)


def _glare(frame, k):
    f = frame.astype(np.float32) * (1.0 + k) + 40 * k
    return np.clip(f, 0, 255).astype(np.uint8)


def _shift(frame, dx, dy):
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _rearrange(frame, box, rng):
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = _clip_box(box, frame.shape)
    rw, rh = w // 6, h // 6
    out = frame.copy()

    def pick():
        for _ in range(60):
            x = rng.randrange(0, max(1, w - rw))
            y = rng.randrange(0, max(1, h - rh))
            if x + rw < bx1 or x > bx2 or y + rh < by1 or y > by2:
                return x, y
        return None

    a, b = pick(), pick()
    if a is None or b is None:
        return out
    (ax, ay), (bx, by) = a, b
    pa = out[ay:ay + rh, ax:ax + rw].copy()
    pb = out[by:by + rh, bx:bx + rw].copy()
    out[ay:ay + rh, ax:ax + rw] = pb
    out[by:by + rh, bx:bx + rw] = pa
    return out


def _paste(frame, patch, box):
    out = frame.copy()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return out
    out[y1:y2, x1:x2] = cv2.resize(patch, (bw, bh))
    return out


PRESENT = {"present": True, "visible": True, "occlusion": "none"}
HIDDEN = {"present": True, "visible": False, "occlusion": "full"}
PARTIAL = {"present": True, "visible": True, "occlusion": "partial"}
ABSENT = {"present": False, "visible": False, "occlusion": "none"}


def build_episodes(frame, box, distractor_patches=None, impostors=None,
                   repeats=REPEATS, seed=SEED):
    eps = {}
    rng = random.Random(seed)

    def episode(base, kind, note, dlen, disrupt_fn, recover_fn, params,
                extra_meta=None):
        name = f"{base}_r{params['repeat']}_d{dlen}"
        frames, gt = [], []
        for _ in range(BASELINE_FRAMES):
            frames.append(frame.copy())
            gt.append(dict(PRESENT))
        for i in range(dlen):
            f, g = disrupt_fn(i)
            frames.append(f)
            gt.append(g)
        for i in range(RECOVER_FRAMES):
            f, g = recover_fn(i)
            frames.append(f)
            gt.append(g)
        meta = {"kind": kind, "note": note,
                "target": TARGET_NAME[0],
                "type": base,
                "disrupt_frames": dlen,
                "disrupt_start": BASELINE_FRAMES,
                "recover_start": BASELINE_FRAMES + dlen,
                "params": params,
                "box": box}
        meta.update(extra_meta or {})
        eps[name] = {"frames": frames, "gt": gt, "meta": meta}

    gone = _remove_target(frame, box)

    def occ_partial(dlen, p):
        frac = p["fraction"]
        col = p["colour"]
        episode("occlusion_partial", "occlusion",
                "half covered, then revealed -- should never be lost", dlen,
                lambda i: (_occlude(frame, box, frac, col, rng), dict(PARTIAL)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def occ_full(dlen, p):
        col = p["colour"]
        episode("occlusion_full", "occlusion",
                "fully hidden then revealed -- correct action is WAIT, not "
                "search", dlen,
                lambda i: (_occlude(frame, box, 1.0, col, rng), dict(HIDDEN)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def occ_remove(dlen, p):
        col = p["colour"]
        episode("occlusion_remove", "occlusion",
                "THE FALSE-BELIEF TEST: removed while hidden. Reporting "
                "'present' after reveal is the failure that matters.", dlen,
                lambda i: (_occlude(frame, box, 1.0, col, rng), dict(HIDDEN)),
                lambda i: (gone.copy(), dict(ABSENT)), p)

    def impostor_ep(dlen, p):
        if not impostors:
            return
        img, shift, dconf = impostors[p["repeat"] % len(impostors)]
        episode("impostor_same_class", "identity",
                "SAME-CLASS IMPOSTOR: swapped while hidden. Geometry says "
                "'unchanged'; only identity verification can reject it.", dlen,
                lambda i: (_occlude(frame, box, 1.0, p["colour"], rng),
                           dict(HIDDEN)),
                lambda i: (img.copy(), dict(ABSENT)), p,
                extra_meta={"impostor_hue_shift": shift,
                            "impostor_detector_conf": round(dconf, 3)})

    def lighting_ep(dlen, p):
        step = p["dim_step"]
        episode("lighting", "environment",
                "dimming ramp -- object never moved, so correct action is "
                "RE-VERIFY IN PLACE, not search", dlen,
                lambda i: (_relight(frame, 1.0 - step * (i + 1)), dict(PRESENT)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def white_balance_ep(dlen, p):
        k = p["wb"]
        episode("white_balance", "environment",
                "colour-temperature shift -- attacks the target's NAMED "
                "attribute while preserving every edge", dlen,
                lambda i: (_white_balance(frame, k * (i + 1) / dlen),
                           dict(PRESENT)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def shadow_ep(dlen, p):
        s, o = p["shadow"], p["shadow_origin"]
        episode("shadow", "environment",
                "soft shadow sweeps over the target -- spatially varying, so "
                "it does not cancel out of region comparisons", dlen,
                lambda i: (_shadow(frame, s * (i + 1) / dlen, o), dict(PRESENT)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def glare_ep(dlen, p):
        k = p["glare"]
        episode("glare", "environment",
                "overexposure to clipping -- a LIGHTING event that destroys "
                "edges, so it looks structurally like motion", dlen,
                lambda i: (_glare(frame, k * (i + 1) / dlen), dict(PRESENT)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def camera_pose_ep(dlen, p):
        sx, sy = p["shift"]
        episode("camera_pose", "environment",
                "ego-motion -- target is still in frame at a NEW position; "
                "correct action is RE-LOCALISE, not scan", dlen,
                lambda i: (_shift(frame, sx * (i + 1), sy * (i + 1)),
                           dict(PRESENT)),
                lambda i: (_shift(frame, sx * dlen, sy * dlen), dict(PRESENT)),
                p)

    def rearrange_ep(dlen, p):
        shuffled = [_rearrange(frame, box, rng) for _ in range(max(1, dlen))]
        episode("scene_rearrange", "environment",
                "background clutter moves, target does not -- the CONTROL for "
                "ego-motion: same global change, opposite correct belief", dlen,
                lambda i: (shuffled[i % len(shuffled)].copy(), dict(PRESENT)),
                lambda i: (shuffled[-1].copy(), dict(PRESENT)), p)

    def distractor_ep(dlen, p):
        if not distractor_patches:
            return
        patch = distractor_patches[p["repeat"] % len(distractor_patches)][1]
        w = box[2] - box[0]
        side = p["side"]
        dbox = [box[0] + side * w, box[1], box[2] + side * w, box[3]]
        episode("distractor", "unexpected",
                "confusable object appears NEXT TO the target -- must not "
                "switch identity", dlen,
                lambda i: (_paste(frame, patch, dbox), dict(PRESENT)),
                lambda i: (frame.copy(), dict(PRESENT)), p)

    def novel_ep(dlen, p):
        if not distractor_patches:
            return
        cls, patch = distractor_patches[p["repeat"] % len(distractor_patches)]
        episode("novel_at_target", "unexpected",
                "target REPLACED by a DIFFERENT-CLASS object in the same "
                "place -- the spurious-re-acquisition trap", dlen,
                lambda i: (_paste(gone, patch, box), dict(ABSENT)),
                lambda i: (_paste(gone, patch, box), dict(ABSENT)),
                p, extra_meta={"replacement_class": cls})

    builders = [occ_partial, occ_full, occ_remove, impostor_ep,
                lighting_ep, white_balance_ep, shadow_ep, glare_ep,
                camera_pose_ep, rearrange_ep, distractor_ep, novel_ep]

    for r in range(repeats):
        for ti, build in enumerate(builders):
            dlen = DISRUPT_LENGTHS[(ti + r) % len(DISRUPT_LENGTHS)]
            p = {
                "repeat": r,
                "fraction": round(rng.uniform(0.35, 0.65), 3),
                "colour": OCCLUDER_COLOURS[rng.randrange(len(OCCLUDER_COLOURS))],
                "dim_step": round(rng.uniform(0.04, 0.09), 3),
                "wb": round(rng.uniform(0.20, 0.45), 3),
                "shadow": round(rng.uniform(0.45, 0.80), 3),
                "shadow_origin": (rng.randrange(frame.shape[1]),
                                  rng.randrange(frame.shape[0])),
                "glare": round(rng.uniform(0.35, 0.80), 3),
                "shift": (rng.choice([-1, 1]) * rng.randrange(12, 24),
                          rng.choice([-1, 1]) * rng.randrange(4, 10)),
                "side": rng.choice([-1, 1]),
            }
            build(dlen, p)
    return eps


def save(eps, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    for name, ep in eps.items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(ep["frames"]):
            cv2.imwrite(os.path.join(d, f"{i:03d}.jpg"), f)
        index[name] = {"dir": d, "n": len(ep["frames"]),
                       "gt": ep["gt"], "meta": ep["meta"]}
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=2)
    return index


def load(out_dir=OUT_DIR):
    idx = json.load(open(os.path.join(out_dir, "index.json")))
    for name, ep in idx.items():
        ep["frames"] = [os.path.join(ep["dir"], f"{i:03d}.jpg")
                        for i in range(ep["n"])]
    return idx


if __name__ == "__main__":
    import shutil
    import sys

    args = list(sys.argv[1:])
    frame_path = None
    repeats = REPEATS
    if "--frame" in args:
        i = args.index("--frame")
        frame_path = args[i + 1]
        del args[i:i + 2]
    if "--repeats" in args:
        i = args.index("--repeats")
        repeats = int(args[i + 1])
        del args[i:i + 2]
    target = " ".join(args).strip() or "red cup"

    if frame_path:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise SystemExit(f"ABORT: cannot read {frame_path}")
        print(f"base frame: {frame_path} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        from camera import Camera
        cam = Camera()
        if not cam.warm:
            cam.release()
            raise SystemExit("ABORT: camera never warmed up")
        ok, frame = cam.read_fresh()
        cam.release()
        if not ok or frame is None:
            raise SystemExit("ABORT: camera read failed")

    TARGET_NAME[0] = target
    found = _target_box(frame, target)
    if not found:
        raise SystemExit(
            f"ABORT: {target!r} not visible -- aim the camera at it first "
            f"(try `yolo {target}` to check)")
    conf, box = found
    print(f"target {target!r} at conf {conf:.2f}, box {box}")

    patches = []
    for d in DISTRACTOR_CLASSES:
        got = _target_box(frame, d)
        if got and got[0] >= DISTRACTOR_MIN_CONF:
            iou = _box_iou(got[1], box)
            if iou > DISTRACTOR_MAX_IOU:
                print(f"  rejecting {d!r} as a distractor: IoU {iou:.3f} with "
                      f"the target -- it IS the target")
                continue
            x1, y1, x2, y2 = got[1]
            patches.append((d, frame[y1:y2, x1:x2].copy()))
            print(f"distractor patch: {d!r} at conf {got[0]:.2f} "
                  f"(IoU {iou:.3f} with target)")
        if len(patches) >= repeats:
            break
    if not patches:
        print("no confusable object in frame -- skipping distractor episodes")

    impostors = build_impostors(frame, box, target)
    if not impostors:
        print("!! NO IMPOSTOR: no hue rotation kept the detector firing on "
              f"{target!r}. The identity episodes will be SKIPPED and D vs E "
              "still cannot test verification. Try a target whose name carries "
              "a colour attribute.")
    else:
        for k, (img, shift, conf) in enumerate(impostors[:repeats]):
            cv2.imwrite(f"_impostor_check_{k}.jpg", img)
            print(f"impostor {k}: hue +{shift} deg, detector still calls it "
                  f"{target!r} at conf {conf:.2f}  -> _impostor_check_{k}.jpg")
        if len(impostors) < repeats:
            print(f"   NOTE only {len(impostors)} distinct impostor(s) for "
                  f"{repeats} repeats -- identity episodes will reuse some.")

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    eps = build_episodes(frame, box, patches, impostors, repeats=repeats)
    index = save(eps)
    print(f"\nwrote {len(index)} episodes to {OUT_DIR}/  "
          f"({sum(e['n'] for e in index.values())} frames total)")

    by_kind = {}
    for name, e in index.items():
        by_kind.setdefault(e["meta"]["kind"], []).append(name)
    for kind, names in sorted(by_kind.items()):
        print(f"  {kind:<12} {len(names):>3} episodes")
    lens = {}
    for e in index.values():
        lens[e["meta"]["disrupt_frames"]] = lens.get(
            e["meta"]["disrupt_frames"], 0) + 1
    print(f"  disrupt lengths: {dict(sorted(lens.items()))}")
    print("\nNEXT: eyeball episodes/*/ before measuring anything against them.")
