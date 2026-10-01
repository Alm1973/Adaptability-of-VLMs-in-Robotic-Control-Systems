import sys
import time

import cv2

from arm import Arm
from camera import Camera
from controller import Controller
from yolo_tracker import (YoloTracker, CONTEXT_CLASSES, UNRELIABLE_CLASSES,
                          MIN_CONFIDENCE)

LOOK_FRAMES = 6
BOUND_MARGIN = 8
SETTLED_FRAMES = 5

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def draw(frame, r, err, move, arm, ctrl, fps, target, status):
    h, w = frame.shape[:2]
    cx0, cy0 = w // 2, h // 2
    dz = ctrl.deadzone

    cv2.rectangle(frame, (cx0 - dz, cy0 - dz), (cx0 + dz, cy0 + dz), (90, 90, 90), 1)
    cv2.drawMarker(frame, (cx0, cy0), GREEN, cv2.MARKER_CROSS, 28, 2)

    if r:
        x, y, bw, bh = r["box"]
        cx, cy = r["center"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), RED, 2)
        cv2.circle(frame, (cx, cy), 5, RED, -1)
        cv2.line(frame, (cx0, cy0), (cx, cy), YELLOW, 1)
        cv2.putText(frame, f"{r['label']} {r['conf']:.2f}", (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    lines = [
        f"target : {target}",
        f"err    : {err[0]:+d},{err[1]:+d}" if err else "err    : --",
        f"move   : {move[0]:+d},{move[1]:+d}" if move else "move   : 0,0",
        f"arm    : {arm.base},{arm.tilt}",
        f"fps    : {fps:.1f}",
        f"status : {status}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "q = quit", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, WHITE, 1, cv2.LINE_AA)
    return frame


def survey(tracker, cam, target):
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    best = 0.0
    for _ in range(LOOK_FRAMES):
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        r = tracker.find(frame)
        if r:
            best = max(best, r["conf"])
    ok, frame = cam.read_fresh()
    others = {}
    if ok and frame is not None:
        tracker.set_targets(CONTEXT_CLASSES, allow_unreliable=True, quiet=True)
        res = tracker.model.predict(frame, conf=0.15, verbose=False)[0]
        for b in res.boxes:
            n = tracker.model.names[int(b.cls.item())]
            others[n] = max(others.get(n, 0.0), float(b.conf.item()))
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    tracker.reset_history()
    return best, others


def at_bound(arm):
    return (arm.base <= BOUND_MARGIN or arm.base >= 180 - BOUND_MARGIN
            or arm.tilt <= BOUND_MARGIN or arm.tilt >= 180 - BOUND_MARGIN)


def follow(tracker, cam, arm, ctrl, target, view, max_seconds, once):
    frames = hits = commands = 0
    settled = 0
    t0 = time.time()
    fps = 0.0
    reason = "quit"
    try:
        while True:
            if max_seconds and time.time() - t0 >= max_seconds:
                reason = "time limit"; break

            ok, frame = cam.read_fresh()
            if not ok or frame is None:
                reason = "camera read failure"; break
            frames += 1
            if frames % 5 == 0:
                fps = frames / max(1e-6, time.time() - t0)

            r = tracker.find(frame)
            err = move = None
            status = "searching"

            if r:
                hits += 1
                h, w = frame.shape[:2]
                cx, cy = r["center"]
                err = (cx - w // 2, cy - h // 2)
                move = ctrl.compute(*err)
                if move == (0, 0):
                    settled += 1
                    status = "CENTRED"
                    if once and settled >= SETTLED_FRAMES:
                        reason = "CENTRED (--once)"
                        if view:
                            cv2.imshow("robot view", draw(frame, r, err, move,
                                                          arm, ctrl, fps, target,
                                                          status))
                            cv2.waitKey(400)
                        break
                else:
                    settled = 0
                    status = "tracking"
                    if at_bound(arm):
                        reason = (f"BOUND GUARD -- joint at limit "
                                  f"(base={arm.base}, tilt={arm.tilt})")
                        status = "AT LIMIT"
                        if view:
                            cv2.imshow("robot view", draw(frame, r, err, move,
                                                          arm, ctrl, fps, target,
                                                          status))
                            cv2.waitKey(800)
                        break
                    try:
                        arm.update(*move)
                    except Exception as e:
                        print(f"[ARM] write FAILED: {e}")
                        reason = "arm write failure"; break
                    commands += 1
            else:
                settled = 0
                ctrl.reset()

            if view:
                cv2.imshow("robot view",
                           draw(frame, r, err, move, arm, ctrl, fps, target,
                                status))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    reason = "quit (q)"; break
                if cv2.getWindowProperty("robot view", cv2.WND_PROP_VISIBLE) < 1:
                    reason = "window closed"; break
    except KeyboardInterrupt:
        reason = "interrupted (Ctrl+C)"
    return reason, frames, hits, commands, time.time() - t0


def main():
    args = [a for a in sys.argv[1:]]
    view = "--no-view" not in args
    once = "--once" in args
    max_seconds = 0.0
    if "--seconds" in args:
        i = args.index("--seconds")
        max_seconds = float(args[i + 1])
        del args[i:i + 2]
    target = " ".join(a for a in args if not a.startswith("--")).strip()
    if not target:
        target = input("What should the arm find? ").strip()
    if not target:
        raise SystemExit("no target given")

    if target in UNRELIABLE_CLASSES:
        print(f"\n!! {target!r} is MEASURED UNRELIABLE here: "
              f"{UNRELIABLE_CLASSES[target]}")
        print("   On held-out data a frame with NO bottle scored HIGHER (0.24)")
        print("   than every frame that had one (0.007-0.118) -- no threshold")
        print("   separates them, so the arm may confidently chase a can.")
        if input("   Track it anyway? [y/N] ").strip().lower() != "y":
            raise SystemExit("cancelled")

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never produced a non-blank frame.")

    tracker = YoloTracker(default_target=target)
    print(f"\n[1/2] LOOKING for {target!r} (no movement)...")
    best, others = survey(tracker, cam, target)

    if best <= 0:
        print(f"\n  {target!r} NOT VISIBLE (nothing above {MIN_CONFIDENCE}).")
        if others:
            print("  What IS visible right now:")
            for n, c in sorted(others.items(), key=lambda x: -x[1])[:6]:
                print(f"    {n:<20} {c:.2f}")
            print("\n  If your target is on the desk, the camera may be aimed")
            print("  too high -- the desk can sit below the frame.")
        else:
            print("  Nothing recognised -- check where the camera points.")
        cam.release()
        return

    print(f"  found at conf {best:.2f}\n")
    arm = Arm()
    start = (arm.base, arm.tilt)
    ctrl = Controller()
    mode = "FOLLOW (continuous)" if not once else "TRACK (stop when centred)"
    print(f"[2/2] {mode}  view={'on' if view else 'off'}"
          + (f"  limit={max_seconds}s" if max_seconds else "")
          + f"  start={start}")
    print("      press q in the window (or Ctrl+C) to stop\n")

    try:
        reason, frames, hits, cmds, el = follow(
            tracker, cam, arm, ctrl, target, view, max_seconds, once)
    finally:
        if view:
            cv2.destroyAllWindows()
        if arm.connection_healthy:
            db, dt = arm.base - start[0], arm.tilt - start[1]
            if db or dt:
                print(f"[ARM] re-homing ({db:+d},{dt:+d}) -> {start}")
                try:
                    arm.update(db, dt)
                except Exception as e:
                    print(f"[ARM] re-home failed: {e}")
        cam.release()

    print("\n" + "=" * 68)
    print(f"target      {target!r}")
    print(f"stopped     {reason}")
    print(f"frames      {frames}   detections {hits}"
          + (f"  ({100.0 * hits / frames:.0f}%)" if frames else ""))
    print(f"commands    {cmds}")
    print(f"elapsed     {el:.1f}s" + (f"  ({frames / el:.1f} fps)" if el else ""))
    print(f"arm         {(arm.base, arm.tilt)} (started {start})")
    print("=" * 68)


if __name__ == "__main__":
    main()
