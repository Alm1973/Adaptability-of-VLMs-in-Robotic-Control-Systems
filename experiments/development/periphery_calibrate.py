import json
import sys
import time

import cv2
import numpy as np

import disruption_bench as db
from camera import Camera
from recovery_pipeline import (_diff, _edge_corr, _periphery_diff,
                               _region_diff)

OUT = "periphery_calibrate.json"
WIN = "PERIPHERY CALIBRATION -- follow the instruction"
SETUP_S = 8.0
RECORD_S = 6.0
ARM_STEP = 10
BOUND_MARGIN = 25


def banner(cam, text, seconds, phase):
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            return True
        ok, f = cam.read()
        if not ok or f is None:
            continue
        h, w = f.shape[:2]
        cv2.rectangle(f, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(f, phase, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 255), 2)
        cv2.putText(f, text[:52], (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(f, f"{left:0.0f}", (w - 90, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.imshow(WIN, f)
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            return False


def sample(cam, ref, box, seconds, label):
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, f = cam.read()
        if not ok or f is None:
            continue
        g = _diff(f, ref)
        l = _region_diff(f, ref, box)
        p = _periphery_diff(f, ref, box)
        e = _edge_corr(f, ref)
        rows.append({"label": label, "global": round(g, 2),
                     "local": round(l, 2), "periphery": round(p, 2),
                     "edge": round(e, 3)})
        hud = f.copy()
        h, w = hud.shape[:2]
        cv2.rectangle(hud, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.putText(hud, f"RECORDING {label}", (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(hud, f"periph {p:6.2f}   global {g:6.2f}", (14, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow(WIN, hud)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            raise KeyboardInterrupt("aborted")
    return rows


def stats(rows, key="periphery"):
    v = sorted(r[key] for r in rows)
    if not v:
        return None
    def pct(q):
        return v[min(len(v) - 1, int(q * len(v)))]
    return {"n": len(v), "min": v[0], "p10": pct(0.10), "p50": v[len(v) // 2],
            "p90": pct(0.90), "max": v[-1],
            "mean": round(sum(v) / len(v), 2)}


PAN_SETTLE_FRAMES = 8


def main():
    with_arm = "--with-arm" in sys.argv
    target = " ".join(a for a in sys.argv[1:]
                      if not a.startswith("-")).strip() or "red cup"

    print("=" * 66)
    print(" PERIPHERY CALIBRATION")
    print(f" arm movement: {'ENABLED' if with_arm else 'disabled'}")
    print("=" * 66)

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")

    ok, ref = cam.read_fresh()
    if not ok or ref is None:
        cam.release()
        raise SystemExit("ABORT: camera read failed")
    found = db._target_box(ref, target)
    if not found:
        cam.release()
        raise SystemExit(f"ABORT: {target!r} not visible")
    conf, box = found
    print(f"target {target!r} conf {conf:.2f} box {box}\n")

    arm = None
    rows = []
    try:
        if not banner(cam, "HANDS OFF. Recording a still baseline.",
                      SETUP_S, "1/3  BASELINE"):
            raise KeyboardInterrupt
        rows += sample(cam, ref, box, RECORD_S, "still")

        if not banner(cam, "COVER the cup with your hand. Hold it there.",
                      SETUP_S, "2/3  HAND OCCLUSION"):
            raise KeyboardInterrupt
        rows += sample(cam, ref, box, RECORD_S, "hand")

        if with_arm:
            if not banner(cam, "HANDS OFF -- the arm will pan by itself.",
                          SETUP_S, "3/3  CAMERA PAN"):
                raise KeyboardInterrupt
            from arm import Arm
            print("opening arm (resets Uno -> centres at 90/90)...")
            arm = Arm()
            time.sleep(2.5)
            ok, ref2 = cam.read_fresh()
            if ok and ref2 is not None:
                ref = ref2
            nb = arm.base + ARM_STEP
            if not (BOUND_MARGIN < nb < 180 - BOUND_MARGIN):
                print("  skipping pan -- would exceed bounds")
            else:
                arm.update(ARM_STEP, 0)
                if not arm.connection_healthy:
                    raise RuntimeError("arm link unhealthy")
                rows += sample(cam, ref, box, RECORD_S, "pan")
        else:
            print("skipping the pan phase (--with-arm not given). Without it "
                  "there is\nno ego-motion cloud to compare against, so no "
                  "threshold can be set.")
    except KeyboardInterrupt as e:
        print(f"\ninterrupted: {e}")
    except Exception as e:
        print(f"\nABORT: {e}")
    finally:
        if arm is not None:
            print("\nreturning arm to centre...")
            try:
                arm.update(arm.base - 90, arm.tilt - 90)
                time.sleep(1.2)
            except Exception as e:
                print(f"  (re-home failed: {e})")
        cam.release()
        cv2.destroyAllWindows()

    json.dump(rows, open(OUT, "w"), indent=2)
    print("\n" + "=" * 66)
    print("PERIPHERY DIFF BY CONDITION")
    print("=" * 66)
    print(f"{'condition':<12}{'n':>5}{'min':>9}{'median':>9}{'max':>9}"
          f"{'mean':>9}")
    print("-" * 53)
    by = {}
    for lab in ("still", "hand", "pan"):
        sel = [r for r in rows if r["label"] == lab]
        if lab == "pan":
            sel = sel[PAN_SETTLE_FRAMES:]
        s = stats(sel)
        by[lab] = s
        if s:
            print(f"{lab:<12}{s['n']:>5}{s['min']:>9.2f}{s['p50']:>9.2f}"
                  f"{s['max']:>9.2f}{s['mean']:>9.2f}")

    print("\nOTHER SIGNALS (median), for diagnosing a failure:")
    for lab in ("still", "hand", "pan"):
        sel = [r for r in rows if r["label"] == lab]
        if sel:
            print(f"  {lab:<8} global {stats(sel,'global')['p50']:>7.2f}   "
                  f"local {stats(sel,'local')['p50']:>7.2f}   "
                  f"edge {stats(sel,'edge')['p50']:>6.3f}")

    h, p = by.get("hand"), by.get("pan")
    print("\n" + "-" * 66)
    if not (h and p):
        print("Need BOTH a hand phase and a pan phase to set a threshold. "
              "Re-run with\n--with-arm, attended.")
        return
    if h["p90"] < p["p10"]:
        lo, hi = h["p90"], p["p10"]
        rec = round(lo + (hi - lo) / 2, 1)
        print(f"SEPARATED: hand p90 = {h['p90']:.2f}, pan p10 = "
              f"{p['p10']:.2f} (medians {h['p50']:.2f} vs {p['p50']:.2f}).")
        print(f"  -> set PERIPHERY_CHANGE = {rec} (midpoint of the gap)")
        print(f"  Extremes for reference: hand max {h['max']:.2f}, "
              f"pan min {p['min']:.2f}.")
    else:
        print(f"OVERLAP: hand p90 = {h['p90']:.2f} vs pan p10 = "
              f"{p['p10']:.2f}.")
        print("  The periphery test does NOT cleanly separate a hand from a "
              "pan on this\n  scene. Do not tune a threshold into the "
              "overlap -- the discriminator\n  needs rethinking. Check the "
              "other signals above for a better one.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
