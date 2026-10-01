import json
import os
import sys

import cv2

FRAMES = "generated_frames"
OUT = "generated_bench.json"

VERIFY_BY_EYE = {
    "present_00.png": True,
    "present_01.png": True,
    "present_02.png": True,
    "present_03.png": True,
    "absent_00.png": False,
    "absent_01.png": False,
    "absent_02.png": False,
    "absent_03.png": False,
}


def load():
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/ -- generate images first")
    rows = []
    for fn in sorted(os.listdir(FRAMES)):
        if not fn.lower().endswith((".png", ".jpg")):
            continue
        if fn.startswith("present"):
            gt = True
        elif fn.startswith("absent"):
            gt = False
        else:
            continue
        rows.append({"file": fn, "gt_prompt": gt,
                     "gt_eye": VERIFY_BY_EYE.get(fn),
                     "path": os.path.join(FRAMES, fn)})
    return rows


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    rows = load()
    if not rows:
        raise SystemExit("ABORT: no images matched present_*/absent_*")

    from yolo_tracker import YoloTracker
    from run_study import make_fast_verifier

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    verifier = make_fast_verifier()

    print(f"target={target!r}  images={len(rows)}\n")
    print(f"{'file':<22}{'GT':>6}{'det':>8}{'conf':>7}{'vlm':>7}  verdict")
    print("-" * 62)

    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            r["error"] = "unreadable"
            continue
        res = tracker.model.predict(img, conf=0.05, verbose=False)[0]
        best = None
        for b in res.boxes:
            name = tracker.model.names[int(b.cls)]
            c = float(b.conf)
            if name == target and (best is None or c > best[0]):
                best = (c, [int(v) for v in b.xyxy[0]])
        r["detected"] = best is not None
        r["conf"] = round(best[0], 3) if best else 0.0
        r["box"] = best[1] if best else None

        ans = verifier(f"Is there a {target} in this image?", img)
        r["vlm"] = str(ans).strip().lower()[:12]
        r["vlm_yes"] = r["vlm"].startswith("y")

        gt = r["gt_eye"] if r["gt_eye"] is not None else r["gt_prompt"]
        r["gt_used"] = gt
        r["det_ok"] = (r["detected"] == gt)
        r["vlm_ok"] = (r["vlm_yes"] == gt)
        mark = "" if r["gt_eye"] is not None else "  (prompt-GT, unverified)"
        print(f"{r['file']:<22}{str(gt):>6}{str(r['detected']):>8}"
              f"{r['conf']:>7.3f}{r['vlm']:>7}  "
              f"{'det:OK' if r['det_ok'] else 'det:MISS'} "
              f"{'vlm:OK' if r['vlm_ok'] else 'vlm:MISS'}{mark}")

    scored = [r for r in rows if "detected" in r]
    eyed = [r for r in scored if r["gt_eye"] is not None]
    print("\n" + "=" * 62)
    print("DOES AVI's PERCEPTION TRANSFER TO GENERATED IMAGERY?")
    print("=" * 62)

    def rate(sel, key):
        return f"{sum(r[key] for r in sel)}/{len(sel)}" if sel else "n/a"

    print(f"  eye-verified GT   n={len(eyed):<3} "
          f"detector {rate(eyed, 'det_ok')}   verifier {rate(eyed, 'vlm_ok')}")
    print(f"  prompt-GT (all)   n={len(scored):<3} "
          f"detector {rate(scored, 'det_ok')}   "
          f"verifier {rate(scored, 'vlm_ok')}")
    if not eyed:
        print("\n  ⚠️ NOTHING is eye-verified yet, so the headline row is empty "
              "by design.\n  The prompt-GT row is PROVISIONAL -- a diffusion "
              "model asked for 'no cup'\n  may well have drawn one. Fill in "
              "VERIFY_BY_EYE before quoting any of\n  these numbers.")

    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
