import glob
import itertools
import json
import statistics as st

import cv2
import numpy as np


def scene_diff(a, b, size=(160, 120)):
    ga = cv2.cvtColor(cv2.resize(a, size), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, size), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def jitter(img, rng):
    h, w = img.shape[:2]
    dx, dy = rng.randint(-1, 2), rng.randint(-1, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    noise = rng.normal(0, 2.0, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    beta = rng.randint(-3, 4)
    return cv2.convertScaleAbs(out, alpha=1.0, beta=beta)


def main():
    paths = sorted(glob.glob("exp_*.jpg")) + sorted(glob.glob("c2_*.jpg"))
    imgs = {p: cv2.imread(p) for p in paths}
    imgs = {p: im for p, im in imgs.items() if im is not None}
    if len(imgs) < 4:
        print("need more frames")
        return
    rng = np.random.RandomState(11)
    print(f"SCENE GATE: {len(imgs)} frames\n")

    static = []
    for p, im in imgs.items():
        for _ in range(4):
            static.append(scene_diff(im, jitter(im, rng)))

    changed = []
    names = list(imgs)
    for a, b in itertools.combinations(names, 2):
        changed.append(scene_diff(imgs[a], imgs[b]))

    c2 = sorted([n for n in names if n.startswith("c2_")])
    adjacent = [scene_diff(imgs[c2[i]], imgs[c2[i + 1]])
                for i in range(len(c2) - 1)]

    print("static (same scene + jitter):")
    print(f"  n={len(static)} mean={st.mean(static):.3f} "
          f"max={max(static):.3f} p95={sorted(static)[int(.95*len(static))]:.3f}")
    print("changed (different frames):")
    print(f"  n={len(changed)} mean={st.mean(changed):.3f} "
          f"min={min(changed):.3f} p05={sorted(changed)[int(.05*len(changed))]:.3f}")
    print("adjacent poses (hardest changed case):")
    print(f"  n={len(adjacent)} mean={st.mean(adjacent):.3f} "
          f"min={min(adjacent):.3f}")

    gap = min(changed) - max(static)
    print(f"\nseparation: max(static)={max(static):.3f}  "
          f"min(changed)={min(changed):.3f}  gap={gap:+.3f}")

    print(f"\n{'thresh':<9} {'skip_static%':<14} {'MISSED_changes':<16} "
          f"{'missed_adjacent'}")
    rows = []
    for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        skip = sum(1 for d in static if d < t) / len(static)
        missed = sum(1 for d in changed if d < t)
        missed_adj = sum(1 for d in adjacent if d < t)
        rows.append({"thresh": t, "skip_static": round(skip, 3),
                     "missed_changed": missed, "missed_adjacent": missed_adj})
        print(f"{t:<9} {skip*100:<14.1f} {missed:<16} {missed_adj}")

    json.dump({"static": static, "changed": changed, "adjacent": adjacent,
               "sweep": rows, "gap": gap},
              open("scene_gate_test.json", "w"), indent=2)

    safe = [r for r in rows if r["missed_changed"] == 0]
    if safe:
        best = max(safe, key=lambda r: r["skip_static"])
        print(f"\nSAFEST USEFUL THRESHOLD: {best['thresh']} -> skips "
              f"{best['skip_static']*100:.0f}% of static frames with "
              f"0 missed changes.")
        print(f"At ~3.2s per detection, a stationary robot would avoid "
              f"~{best['skip_static']*100:.0f}% of that cost.")
    else:
        print("\nNO SAFE THRESHOLD -- static and changed distributions "
              "overlap; scene gating is NOT viable on this data.")


if __name__ == "__main__":
    main()
