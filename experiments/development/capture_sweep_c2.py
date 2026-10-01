
import json
import time

import cv2

from arm import Arm
from camera import Camera

BOUND_LO, BOUND_HI = 15, 165
SETTLE_READS = 18
SETTLE_SLEEP = 0.06
FRAME_DIFF_STUCK = 4.0
MOVE_PAUSE = 0.8

POSES = []
for b in range(60, 121, 12):
    POSES.append((b, 90, f"pan_b{b}_t90"))
for t in range(78, 109, 15):
    POSES.append((90, t, f"tilt_b90_t{t}"))
POSES.append((90, 90, "home"))


def clamp(a):
    return max(BOUND_LO, min(BOUND_HI, a))


def settle_and_grab(cam, reads=SETTLE_READS):
    f = None
    for _ in range(reads):
        ok, f = cam.read()
        time.sleep(SETTLE_SLEEP)
    return f


def frame_diff(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def main():
    print("Opening camera (settling AE)...")
    cam = Camera()
    time.sleep(1.0)
    settle_and_grab(cam, 40)

    print("Opening arm (opening COM5 resets the Arduino -> arm homes 90/90)...")
    arm = Arm()

    log = []
    prev = None
    aborted = None
    for i, (b, t, label) in enumerate(POSES, 1):
        b, t = clamp(b), clamp(t)
        db, dt_ = b - arm.base, t - arm.tilt
        print(f"[{time.strftime('%H:%M:%S')}] pose {i}/{len(POSES)} {label} "
              f"-> ({b},{t}) d=({db},{dt_})")
        if db or dt_:
            try:
                arm.update(db, dt_)
            except Exception as e:
                aborted = f"ARM_WRITE_FAILED {label}: {e}"
                print("ABORT:", aborted)
                break
            if not arm.connection_healthy:
                aborted = f"ARM_UNHEALTHY {label}"
                print("ABORT:", aborted)
                break
            time.sleep(MOVE_PAUSE)

        frame = settle_and_grab(cam)
        if frame is None:
            aborted = f"NO_FRAME {label}"
            print("ABORT:", aborted)
            break
        if (db or dt_) and prev is not None:
            d = round(frame_diff(prev, frame), 2)
            print(f"  frame diff vs prev: {d}")
            if d < FRAME_DIFF_STUCK:
                aborted = f"STUCK_ARM {label}: cmd({db},{dt_}) but diff {d}"
                print("ABORT:", aborted)
                break

        fn = f"c2_{i:02d}_{label}.jpg"
        cv2.imwrite(fn, frame)
        log.append({"i": i, "label": label, "base": b, "tilt": t,
                    "file": fn, "frame_mean": round(float(frame.mean()), 1)})
        prev = frame

    cam.release()
    json.dump({"aborted": aborted, "poses": len(log), "log": log},
              open("capture_sweep_c2.json", "w"), indent=2)
    print(f"\nposes captured: {len(log)}  aborted: {aborted}")
    print("wrote capture_sweep_c2.json")


if __name__ == "__main__":
    main()
