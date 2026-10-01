import json
import os
import re
import sys

import cv2
import numpy as np

FRAMES = "live_disruption_frames"
OUT = "motion_occluder.json"
SETTLE = 8

CASES = [
    ("occlusion_full", "disrupt", True, "a real hand covering the cup"),
    ("occlusion_remove", "disrupt", True, "a real hand covering the cup"),
    ("impostor", "recover", False, "an object placed where the cup was"),
    ("occlusion_remove", "recover", False, "cup gone, bare desk"),
    ("lighting", "disrupt", None, "control: cup visible, light changing"),
]


def frames_for(scenario, phase):
    out = []
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_{phase}_(\d+)\.jpg$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(FRAMES, fn)))
    return [p for _, p in sorted(out)]


def region_motion(paths, box, settle=SETTLE):
    x1, y1, x2, y2 = box
    prev = None
    inside, outside = [], []
    for p in paths[settle:]:
        img = cv2.imread(p)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = cv2.absdiff(g, prev).astype(np.float32)
            box_d = d[max(0, y1):y2, max(0, x1):x2]
            mask = np.ones(d.shape, bool)
            mask[max(0, y1):y2, max(0, x1):x2] = False
            if box_d.size and mask.sum() > 64:
                inside.append(float(box_d.mean()))
                outside.append(float(d[mask].mean()))
        prev = g
    if not inside:
        return None
    mi = float(np.median(inside))
    mo = float(np.median(outside))
    return {"in": round(mi, 3), "out": round(mo, 3),
            "ratio": round(mi / mo, 3) if mo > 1e-6 else None,
            "n": len(inside)}


def main():
    import disruption_bench as db

    target = " ".join(sys.argv[1:]).strip() or "red cup"
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/")

    boxes = {}
    for scenario in {c[0] for c in CASES}:
        base = frames_for(scenario, "baseline")
        if not base:
            continue
        img = cv2.imread(base[len(base) // 2])
        got = db._target_box(img, target)
        if got:
            boxes[scenario] = got[1]
    if not boxes:
        raise SystemExit("ABORT: could not locate the target in any baseline")

    rows = {}
    print(f"{'case':<34}{'in':>8}{'out':>8}{'ratio':>9}  truth")
    print("-" * 70)
    for scenario, phase, is_occ, desc in CASES:
        if scenario not in boxes:
            continue
        paths = frames_for(scenario, phase)
        if len(paths) <= SETTLE + 2:
            continue
        m = region_motion(paths, boxes[scenario])
        if not m:
            continue
        rows[f"{scenario}/{phase}"] = dict(m, is_occluder=is_occ, desc=desc)
        truth = ("OCCLUDER" if is_occ else
                 "replacement" if is_occ is False else "control")
        print(f"{scenario + '/' + phase:<34}{m['in']:>8.2f}{m['out']:>8.2f}"
              f"{str(m['ratio']):>9}  {truth}")

    occ = [r["ratio"] for r in rows.values()
           if r["is_occluder"] is True and r["ratio"] is not None]
    rep = [r["ratio"] for r in rows.values()
           if r["is_occluder"] is False and r["ratio"] is not None]

    print("\n" + "=" * 70)
    print("DOES MOTION SEPARATE A LIVE HAND FROM A PLACED OBJECT?")
    print("=" * 70)
    if not occ or not rep:
        print("  not enough cases to compare")
    else:
        print(f"  occluder (hand)    ratios: {occ}")
        print(f"  replacement/static ratios: {rep}")
        gap = min(occ) - max(rep)
        spread = max(max(occ) - min(occ), max(rep) - min(rep), 1e-6)
        if min(occ) > max(rep) * 1.5 and gap > spread:
            thr = round((min(occ) + max(rep)) / 2, 2)
            print(f"\n  SEPARATED: hand ratios ({min(occ):.2f}+) exceed static "
                  f"ones ({max(rep):.2f}-)\n  by {gap:.2f}, larger than the "
                  f"within-class spread of {spread:.2f}. A threshold near\n  "
                  f"{thr} decides occluded-vs-replaced for ~1 ms of OpenCV "
                  f"instead of a ~1 s\n  VLM call, and works with NO VLM at "
                  f"all.")
        elif min(occ) > max(rep):
            print(f"\n  NOT SEPARATED -- the ordering is right but the margin "
                  f"is {gap:.3f} against a\n  within-class spread of "
                  f"{spread:.3f}, at n={len(occ)} and n={len(rep)}. That is "
                  f"noise, not\n  a discriminator. Every ratio sits near 1.0, "
                  f"i.e. motion inside the target\n  region tracks motion "
                  f"everywhere else: a held hand is NOT measurably more\n  "
                  f"mobile than a placed object at this frame rate. REJECT "
                  f"the idea and keep\n  the VLM probe.")
        else:
            print(f"\n  OVERLAP: hands go down to {min(occ):.2f} while static "
                  f"regions reach\n  {max(rep):.2f}. Motion alone does not "
                  f"decide this. Keep the VLM probe; the\n  idea is cheap but "
                  f"the signal is not there on this data.")
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
