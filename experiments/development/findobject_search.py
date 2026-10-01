import argparse
import sys
import time

import cv2

from arm import Arm
from camera import Camera
from open_vocab_detect import VOTE_SAMPLES, detect

BOUND_LO, BOUND_HI = 15, 165
HOME = (90, 90)

SETTLE_READS = 16
SETTLE_SLEEP = 0.06
MOVE_PAUSE = 0.7
FRAME_DIFF_STUCK = 3.0

BASE_ANGLES = [45, 63, 81, 99, 117, 135]
TILT_ANGLES = [70, 90, 110]


WINDOW = "robot view - searching"
_show_enabled = True


def show(frame, lines=(), color=(0, 255, 0)):
    global _show_enabled
    if not _show_enabled or frame is None:
        return True
    disp = frame.copy()
    h = 26 * len(lines) + 12 if lines else 0
    if h:
        cv2.rectangle(disp, (0, 0), (disp.shape[1], h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(disp, line, (12, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    try:
        cv2.imshow(WINDOW, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            return False
    except Exception:
        _show_enabled = False
    return True


def clamp(a):
    return max(BOUND_LO, min(BOUND_HI, a))


def frame_diff(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (160, 120)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


class Quit(Exception):
    pass


def settle(cam, status=()):
    f = None
    for _ in range(SETTLE_READS):
        ok, cur = cam.read()
        if ok and cur is not None:
            f = cur
            if not show(f, status):
                raise Quit()
        time.sleep(SETTLE_SLEEP)
    return f


def build_poses():
    poses = []
    for i, t in enumerate(TILT_ANGLES):
        row = BASE_ANGLES if i % 2 == 0 else list(reversed(BASE_ANGLES))
        for b in row:
            poses.append((clamp(b), clamp(t)))
    return poses


def goto(arm, cam, base, tilt, prev_frame, status=()):
    db, dt = arm.base - base, arm.tilt - tilt
    if db or dt:
        try:
            arm.update(db, dt)
        except Exception as e:
            return None, f"ARM_WRITE_FAILED: {e}"
        if not arm.connection_healthy:
            return None, "ARM_UNHEALTHY"
        time.sleep(MOVE_PAUSE)
    frame = settle(cam, status)
    if frame is None:
        return None, "NO_FRAME"
    if (db or dt) and prev_frame is not None:
        d = frame_diff(prev_frame, frame)
        if d < FRAME_DIFF_STUCK:
            return frame, f"STUCK_ARM (moved ({db},{dt}) but frame diff {d:.1f})"
    return frame, None


def search(arm, cam, queries, passes):
    poses = build_poses()
    print(f"\nsearch grid: {len(poses)} poses "
          f"(base {BASE_ANGLES[0]}-{BASE_ANGLES[-1]}, "
          f"tilt {TILT_ANGLES[0]}-{TILT_ANGLES[-1]}), up to {passes} passes\n")
    prev = None
    for p in range(1, passes + 1):
        print(f"--- pass {p}/{passes} ---")
        for i, (b, t) in enumerate(poses, 1):
            status = [f"searching: {', '.join(queries)}",
                      f"pass {p}/{passes}  pose {i}/{len(poses)}  "
                      f"base={b} tilt={t}"]
            frame, abort = goto(arm, cam, b, t, prev, status)
            if abort:
                print(f"  ABORT: {abort}")
                show(frame, [f"ABORT: {abort}"], (0, 0, 255))
                cv2.waitKey(1200)
                return None, abort
            prev = frame
            show(frame, status + ["analyzing..."], (0, 200, 255))
            results, lists = detect(frame, queries, samples=VOTE_SAMPLES)
            found = [q for q in queries if results.get(q)]
            seen = sorted({it for l in lists for it in l}, key=str.lower)
            print(f"  [{i:2d}/{len(poses)}] base={b:3d} tilt={t:3d}  "
                  f"saw: {', '.join(seen) if seen else '-'}")
            if found:
                print(f"\n  ✅ FOUND {found} at base={b}, tilt={t}")
                show(frame, [f"FOUND: {', '.join(found)}",
                             f"base={b} tilt={t}"], (0, 255, 0))
                cv2.waitKey(2500)
                return (b, t, found), None
            if not show(frame, status + [f"saw: {', '.join(seen[:5])}"]):
                raise Quit()
        print(f"  pass {p}: not found\n")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*", help="object name(s) to find")
    ap.add_argument("--passes", type=int, default=3,
                    help="full sweeps before giving up (default 3)")
    ap.add_argument("--no-window", action="store_true",
                    help="disable the live view window")
    args = ap.parse_args()

    global _show_enabled
    if args.no_window:
        _show_enabled = False

    queries = [q.strip() for q in args.queries if q.strip()]
    if not queries:
        try:
            raw = input("what should I search for? > ").strip()
        except EOFError:
            return
        queries = [q.strip() for q in raw.split(",") if q.strip()]
    if not queries:
        return

    print("=" * 60)
    print(" ACTIVE OBJECT SEARCH -- THIS MOVES THE ARM")
    print(" A human should be watching. Ctrl+C stops and returns home.")
    print("=" * 60)
    print(f" looking for: {queries}")

    cam = Camera()
    settle(cam)
    print("\nopening arm (this RESETS the Arduino -> it jumps to 90/90)...")
    arm = Arm()
    time.sleep(1.0)

    result = abort = None
    try:
        result, abort = search(arm, cam, queries, args.passes)
    except KeyboardInterrupt:
        print("\ninterrupted by user")
    except Quit:
        print("\nstopped from the live window (q/ESC)")
    finally:
        print("\nreturning home...")
        try:
            arm.update(HOME[0] - arm.base, HOME[1] - arm.tilt)
            time.sleep(MOVE_PAUSE)
        except Exception as e:
            print(f"  (could not return home: {e})")
        cam.release()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    print()
    if result:
        b, t, found = result
        print(f"RESULT: found {found} at base={b}, tilt={t}")
    elif abort:
        print(f"RESULT: aborted -- {abort}")
    else:
        print(f"RESULT: {queries} not found after {args.passes} passes.")
        print("        Try more passes, or move the object into better light "
              "-- transparent/small objects are the known weak spot.")


if __name__ == "__main__":
    main()
