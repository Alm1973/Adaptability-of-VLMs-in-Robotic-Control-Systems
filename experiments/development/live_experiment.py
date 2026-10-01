import json
import os
import time

import cv2

import llm_backend
from adaptive_detector import (Suppression, configure_validated,
                               detect_candidates, verify)
from arm import Arm
from camera import Camera
from open_vocab_detect import VOTE_SAMPLES, detect as vlm_detect

BOUND_LO, BOUND_HI = 15, 165
HOME = (90, 90)
SETTLE_READS = 16
SETTLE_SLEEP = 0.06
MOVE_PAUSE = 0.7
FRAME_DIFF_STUCK = 3.0

TARGETS = ["keyboard", "monitor", "celsius can"]
POSES = [(70, 95), (90, 95), (110, 95), (110, 110), (90, 110), (70, 110)]
RUNS = 2
OUTDIR = "live_exp_frames"


def clamp(a):
    return max(BOUND_LO, min(BOUND_HI, a))


def settle(cam):
    f = None
    for _ in range(SETTLE_READS):
        ok, cur = cam.read()
        if ok and cur is not None:
            f = cur
        time.sleep(SETTLE_SLEEP)
    return f


def frame_diff(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def goto(arm, cam, base, tilt, prev):
    db, dt = arm.base - base, arm.tilt - tilt
    if db or dt:
        try:
            arm.update(db, dt)
        except Exception as e:
            return None, f"ARM_WRITE_FAILED: {e}"
        if not arm.connection_healthy:
            return None, "ARM_UNHEALTHY"
        time.sleep(MOVE_PAUSE)
    f = settle(cam)
    if f is None:
        return None, "NO_FRAME"
    if (db or dt) and prev is not None and frame_diff(prev, f) < FRAME_DIFF_STUCK:
        return f, "STUCK_ARM"
    return f, None


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("=" * 64)
    print(" LIVE SERVO EXPERIMENT -- THIS MOVES THE ARM (attended only)")
    print("=" * 64)
    print(f" targets={TARGETS}\n poses={POSES}\n runs={RUNS}\n")

    cam = Camera()
    settle(cam)
    print("opening arm (RESETS Arduino -> jumps to 90/90)...")
    arm = Arm()
    time.sleep(1.0)

    records = []
    aborted = None
    prev = None
    try:
        for run in range(1, RUNS + 1):
            for target in TARGETS:
                f0, ab = goto(arm, cam, *HOME, prev)
                prev = f0
                if ab:
                    aborted = ab
                    raise RuntimeError(ab)
                cfg, cfg_time = configure_validated(f0, target, verbose=False)
                cues = cfg.cues if cfg else []
                colour = cfg.color_name if cfg else ""
                print(f"\n[run {run}] target={target!r} cfg={cfg_time:.2f}s "
                      f"cues={cues} colour={colour!r} "
                      f"{'(config failed)' if cfg is None else ''}")

                supp = Suppression()
                for (b, t) in POSES:
                    frame, ab = goto(arm, cam, clamp(b), clamp(t), prev)
                    if ab:
                        aborted = ab
                        raise RuntimeError(ab)
                    prev = frame

                    t0 = time.time()
                    cands = detect_candidates(frame, cfg, suppression=supp) \
                        if cfg else []
                    cv_ms = (time.time() - t0) * 1000
                    new_pred, vcalls, vsecs = 0, 0, 0.0
                    for c in cands[:2]:
                        ok_v, vdt = verify(frame, c["box"], target)
                        vcalls += 1
                        vsecs += vdt
                        if ok_v:
                            new_pred = 1
                            break
                        supp.add(c["center"])
                    new_total = cv_ms / 1000 + vsecs

                    t0 = time.time()
                    bres, _ = vlm_detect(frame, [target], samples=VOTE_SAMPLES)
                    b_secs = time.time() - t0
                    base_pred = int(bool(bres[target]))

                    fname = f"{OUTDIR}/r{run}_{target.replace(' ','_')}_b{b}_t{t}.jpg"
                    cv2.imwrite(fname, frame)
                    records.append({
                        "run": run, "target": target, "base": b, "tilt": t,
                        "frame": fname, "cues": cues, "colour": colour,
                        "config_s": round(cfg_time, 3),
                        "new_opencv_ms": round(cv_ms, 2),
                        "new_verify_calls": vcalls,
                        "new_verify_s": round(vsecs, 3),
                        "new_total_s": round(new_total, 3),
                        "new_pred": new_pred,
                        "baseline_s": round(b_secs, 3),
                        "baseline_calls": VOTE_SAMPLES,
                        "baseline_pred": base_pred,
                        "agree": int(new_pred == base_pred),
                    })
                    print(f"   b={b:3d} t={t:3d}  new {cv_ms:5.1f}ms"
                          f"+{vsecs:4.2f}s pred={new_pred} | "
                          f"base {b_secs:5.2f}s pred={base_pred} | "
                          f"agree={records[-1]['agree']}")
                    json.dump(records, open("live_experiment.json", "w"),
                              indent=2)
    except RuntimeError as e:
        print(f"\nABORT: {e}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\nreturning home...")
        try:
            arm.update(HOME[0] - arm.base, HOME[1] - arm.tilt)
            time.sleep(MOVE_PAUSE)
        except Exception as e:
            print(f" (home failed: {e})")
        cam.release()
        llm_backend.shutdown()

    json.dump(records, open("live_experiment.json", "w"), indent=2)
    print(f"\nwrote {len(records)} records -> live_experiment.json")
    print(f"frames saved to {OUTDIR}/ (label them to compute real accuracy)")
    if aborted:
        print(f"ABORTED: {aborted}")


if __name__ == "__main__":
    main()
