import json
import os
import re
import sys

import cv2
import numpy as np

from recovery_pipeline import _region_vs_surround

FRAMES = "live_disruption_frames"
OUT = "surround_real.json"
SETTLE = 8

CASES = [
    ("occlusion_full", "disrupt", "covered", "hand covering the cup"),
    ("occlusion_remove", "disrupt", "covered", "hand covering the cup"),
    ("occlusion_remove", "recover", "gone", "cup removed, bare desk"),
    ("impostor", "recover", "gone", "object placed where the cup was"),
    ("lighting", "disrupt", "present", "control: cup visible"),
]


def frames_for(scenario, phase):
    out = []
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_{phase}_(\d+)\.jpg$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(FRAMES, fn)))
    return [p for _, p in sorted(out)]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    import disruption_bench as db
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/")

    boxes = {}
    for scenario in {c[0] for c in CASES}:
        base = frames_for(scenario, "baseline")
        if base:
            got = db._target_box(cv2.imread(base[len(base) // 2]), target)
            if got:
                boxes[scenario] = got[1]

    rows, by_class = {}, {}
    print(f"{'case':<34}{'n':>5}{'p10':>8}{'median':>9}{'p90':>8}  truth")
    print("-" * 72)
    for scenario, phase, cls, desc in CASES:
        if scenario not in boxes:
            continue
        sims = []
        for p in frames_for(scenario, phase)[SETTLE:]:
            img = cv2.imread(p)
            if img is None:
                continue
            sim, _ = _region_vs_surround(img, boxes[scenario])
            sims.append(sim)
        if not sims:
            continue
        rows[f"{scenario}/{phase}"] = {
            "class": cls, "desc": desc, "n": len(sims),
            "p10": round(pct(sims, 0.10), 3),
            "median": round(float(np.median(sims)), 3),
            "p90": round(pct(sims, 0.90), 3)}
        r = rows[f"{scenario}/{phase}"]
        by_class.setdefault(cls, []).extend(sims)
        print(f"{scenario + '/' + phase:<34}{r['n']:>5}{r['p10']:>8.3f}"
              f"{r['median']:>9.3f}{r['p90']:>8.3f}  {cls}")

    print("\n" + "=" * 72)
    print("WHERE SHOULD SURROUND_SIMILAR SIT ON REAL FRAMES?")
    print("=" * 72)
    cov, gone = by_class.get("covered", []), by_class.get("gone", [])
    if not cov or not gone:
        print("  need both a covered and a gone class")
    else:
        cov_hi, gone_lo = pct(cov, 0.90), pct(gone, 0.10)
        print(f"  covered  n={len(cov):<4} median "
              f"{np.median(cov):.3f}   p90 {cov_hi:.3f}  (want BELOW thr)")
        print(f"  gone     n={len(gone):<4} median "
              f"{np.median(gone):.3f}   p10 {gone_lo:.3f}  (want ABOVE thr)")
        if cov_hi < gone_lo:
            rec = round((cov_hi + gone_lo) / 2, 2)
            print(f"\n  SEPARATED -> SURROUND_SIMILAR = {rec}")
            print(f"  Current value is 0.40, swept on SYNTHETIC occluders "
                  f"where the tradeoff\n  never fired. This one is measured on "
                  f"real hands.")
        else:
            print(f"\n  OVERLAP: covered reaches {cov_hi:.3f} while gone drops "
                  f"to {gone_lo:.3f}.")
            print("  Interior-vs-background similarity does NOT separate a "
                  "covered target from\n  a removed one on real frames -- a "
                  "hand over a cup can look as much like\n  the desk as the "
                  "bare desk does. No threshold fixes this; it is why the\n"
                  "  VLM region probe exists, and it explains the 53 lost "
                  "frames in the replay.")
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
