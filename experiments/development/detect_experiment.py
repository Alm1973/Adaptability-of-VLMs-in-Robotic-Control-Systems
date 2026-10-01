
import json
import time

import cv2

import detect_methods as dm
from arm import Arm
from camera import Camera

BOUND_MARGIN = 5
SETTLE_READS = 18
SETTLE_SLEEP = 0.06
FRAME_DIFF_STUCK = 4.0

POSES = [
    (90, 90, "home_ref"),
    (98, 90, "verify_small"),
    (80, 90, "pan_left_10"),
    (70, 90, "pan_left_20"),
    (110, 90, "pan_right_20"),
    (90, 78, "tilt_up_12"),
    (90, 102, "tilt_down_12"),
    (78, 80, "diag_upleft"),
    (104, 100, "diag_downright"),
    (68, 102, "hard_lowleft"),
    (114, 78, "hard_highright"),
    (90, 90, "return_home"),
]


def clamp(a):
    return max(BOUND_MARGIN, min(180 - BOUND_MARGIN, a))


def settle_and_grab(cam, reads=SETTLE_READS):
    f = None
    for _ in range(reads):
        ret, f = cam.read()
        time.sleep(SETTLE_SLEEP)
    return f


def frame_diff(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def main():
    template = cv2.imread("template_bottle.jpg")
    if template is None:
        print("FATAL: template_bottle.jpg missing")
        return
    ctx = dm.build_context(template)
    print(f"template ORB keypoints: {ctx['template_kp_count']}")

    print("Opening camera (settling)...")
    cam = Camera()
    time.sleep(1.0)
    settle_and_grab(cam, 40)

    print("Opening arm...")
    arm = Arm()

    log = []
    prev_frame = None
    aborted = None

    for idx, (b, t, label) in enumerate(POSES, start=1):
        b, t = clamp(b), clamp(t)
        db, dt_ = b - arm.base, t - arm.tilt
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] pose {idx}/{len(POSES)} '{label}' -> ({b},{t}) delta=({db},{dt_})")

        if db or dt_:
            try:
                arm.update(db, dt_)
            except Exception as e:
                aborted = f"ARM_WRITE_FAILED at {label}: {e}"
                print(f"  ABORT: {aborted}")
                break
            if not arm.connection_healthy:
                aborted = f"ARM_UNHEALTHY at {label}"
                print(f"  ABORT: {aborted}")
                break
            time.sleep(0.8)

        frame = settle_and_grab(cam)
        if frame is None:
            aborted = f"NO_FRAME at {label}"
            break

        moved = bool(db or dt_)
        diff = None
        if moved and prev_frame is not None:
            diff = round(frame_diff(prev_frame, frame), 2)
            print(f"  frame diff vs prev: {diff}")
            if diff < FRAME_DIFF_STUCK:
                aborted = (f"STUCK_ARM_SIGNATURE at {label}: commanded "
                           f"({db},{dt_}) but frame diff {diff}")
                print(f"  ABORT: {aborted}")
                break

        fname = f"exp_{idx:02d}_{label}_b{b}_t{t}.jpg"
        cv2.imwrite(fname, frame)

        entry = {"idx": idx, "label": label, "base": b, "tilt": t,
                 "file": fname, "frame_mean": round(float(frame.mean()), 1),
                 "frame_diff_vs_prev": diff, "techniques": {}}

        for name, fn in dm.CLASSICAL.items():
            t0 = time.time()
            r = fn(frame, ctx)
            r["latency_ms"] = round((time.time() - t0) * 1000, 1)
            entry["techniques"][name] = r
        ens = dm.t6_ensemble(frame, ctx, precomputed=entry["techniques"])
        entry["techniques"]["t6"] = ens

        print("   " + "  ".join(
            f"{k}={'Y' if v['found'] else 'n'}({v['confidence']:.2f})"
            for k, v in entry["techniques"].items()))
        log.append(entry)
        prev_frame = frame

    cam.release()

    out = {"aborted": aborted, "poses_completed": len(log), "log": log}
    with open("experiment_log.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nposes completed: {len(log)}  aborted: {aborted}")
    print("wrote experiment_log.json")


if __name__ == "__main__":
    main()
