import random
import sys
import time

import cv2

_rng = random.Random()

from arm import Arm
from camera import Camera
from controller import Controller
from recovery_pipeline import (RecoveryPipeline, CONFIRMED, OCCLUDED,
                               DISPLACED, MISSING, AMBIGUOUS)
from yolo_tracker import YoloTracker

BOUND_MARGIN = 8
DECAY_FRAMES = 8

SEARCH_AFTER = 6
SEARCH_STEP = 12
SEARCH_COOLDOWN = 8
MAX_SEARCH_MOVES = 12
MAX_PER_DIRECTION = 2

EXPLORE_EPSILON = 0.4


SEARCH_DELTA = {"right": (SEARCH_STEP, 0), "left": (-SEARCH_STEP, 0),
                "down": (0, SEARCH_STEP), "up": (0, -SEARCH_STEP)}

COLOUR = {CONFIRMED: (0, 220, 0), OCCLUDED: (0, 200, 255),
          DISPLACED: (255, 170, 0), MISSING: (0, 0, 235),
          AMBIGUOUS: (200, 200, 200)}


def draw(frame, pipe, err, move, arm, ctrl, fps, target, note):
    h, w = frame.shape[:2]
    cx0, cy0 = w // 2, h // 2
    dz = ctrl.deadzone
    col = COLOUR.get(pipe.status, (255, 255, 255))

    cv2.rectangle(frame, (cx0 - dz, cy0 - dz), (cx0 + dz, cy0 + dz),
                  (90, 90, 90), 1)
    cv2.drawMarker(frame, (cx0, cy0), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)

    if pipe.box:
        x1, y1, x2, y2 = pipe.box
        thick = 2 if pipe.status == CONFIRMED else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        if pipe.status == CONFIRMED and err:
            cv2.line(frame, (cx0, cy0), ((x1 + x2) // 2, (y1 + y2) // 2),
                     (0, 255, 255), 1)

    cv2.rectangle(frame, (0, 0), (330, 152), (0, 0, 0), -1)
    lines = [
        f"target : {target}",
        f"belief : {pipe.status}",
        f"err    : {err[0]:+d},{err[1]:+d}" if err else "err    : --",
        f"move   : {move[0]:+d},{move[1]:+d}" if move else "move   : hold",
        f"arm    : {arm.base},{arm.tilt}   fps {fps:.1f}",
        f"vlm    : {pipe.vlm_calls} calls",
    ]
    for i, t in enumerate(lines):
        c = col if i == 1 else (255, 255, 255)
        cv2.putText(frame, t, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, c, 2 if i == 1 else 1, cv2.LINE_AA)
    if note:
        cv2.putText(frame, note, (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    cv2.putText(frame, "q = quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def make_verifier():
    from run_study import make_fast_verifier
    return make_fast_verifier()


def at_bound(arm):
    return (arm.base <= BOUND_MARGIN or arm.base >= 180 - BOUND_MARGIN
            or arm.tilt <= BOUND_MARGIN or arm.tilt >= 180 - BOUND_MARGIN)


def _would_exceed(arm, dx, dy):
    nb, nt = arm.base + dx, arm.tilt + dy
    return not (BOUND_MARGIN < nb < 180 - BOUND_MARGIN
                and BOUND_MARGIN < nt < 180 - BOUND_MARGIN)


SEARCH_POLICY = "random"


def choose_direction(pipe, frame, arm, tried):
    legal = [d for d, (dx, dy) in SEARCH_DELTA.items()
             if tried[d] < MAX_PER_DIRECTION and not _would_exceed(arm, dx, dy)]
    if not legal:
        return None, "all directions exhausted or blocked by limits"

    if SEARCH_POLICY == "random":
        return _rng.choice(legal), "random (measured best)"

    if SEARCH_POLICY == "vlm_eps" and _rng.random() < EXPLORE_EPSILON:
        return _rng.choice(legal), "explore"

    want = pipe.search_direction(frame, tried)
    if want in legal:
        return want, "vlm"
    if want is None:
        best = min(legal, key=lambda d: tried[d])
        return best, "vlm gave no answer -> least-tried"
    best = min(legal, key=lambda d: tried[d])
    reason = ("exhausted" if tried.get(want, 0) >= MAX_PER_DIRECTION
              else "would hit limit")
    return best, f"vlm said {want} ({reason}) -> {best}"


def main():
    args = [a for a in sys.argv[1:]]
    view = "--no-view" not in args
    max_seconds = 0.0
    if "--seconds" in args:
        i = args.index("--seconds")
        max_seconds = float(args[i + 1]); del args[i:i + 2]
    target = " ".join(a for a in args if not a.startswith("--")).strip()
    if not target:
        target = input("What should the arm find? ").strip()
    if not target:
        raise SystemExit("no target given")

    cam = Camera()
    if not cam.warm:
        cam.release(); raise SystemExit("ABORT: camera never warmed up")

    print(f"target {target!r} -- loading detector + verifier...")
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    t0 = time.time()
    verifier = make_verifier()
    print(f"  ready in {time.time() - t0:.0f}s")

    from clip_classifier import make_clip_classifier
    pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                            decay_frames=DECAY_FRAMES,
                            use_region_probe=True,
                            probe_verifier=getattr(verifier, "generate", None),
                            probe_classifier=make_clip_classifier())
    ctrl = Controller()
    arm = Arm()
    start = (arm.base, arm.tilt)
    print(f"start pos {start} -- q in the window (or Ctrl+C) to stop\n")

    frames = cmds = 0
    fps = 0.0
    reason = "quit"
    missing_run = 0
    search_moves = 0
    last_search_frame = -999
    tried = {d: 0 for d in SEARCH_DELTA}
    t0 = time.time()
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

            rec = pipe.step(frame)
            err = move = None
            note = rec.get("evidence", {}).get("reason", "")

            if pipe.status == CONFIRMED and pipe.box:
                x1, y1, x2, y2 = pipe.box
                h, w = frame.shape[:2]
                err = ((x1 + x2) // 2 - w // 2, (y1 + y2) // 2 - h // 2)
                move = ctrl.compute(*err)
                if move != (0, 0):
                    if at_bound(arm):
                        reason = (f"BOUND GUARD -- joint at limit "
                                  f"({arm.base},{arm.tilt})")
                        break
                    try:
                        arm.update(*move)
                    except Exception as e:
                        print(f"[ARM] write FAILED: {e}")
                        reason = "arm write failure"; break
                    cmds += 1
                    pipe.notify_self_motion()
            else:
                ctrl.reset()

            if pipe.status == CONFIRMED:
                missing_run = 0
                search_moves = 0
                tried = {d: 0 for d in SEARCH_DELTA}
            elif pipe.status == MISSING:
                missing_run += 1
            else:
                missing_run = 0

            if (missing_run >= SEARCH_AFTER
                    and frames - last_search_frame >= SEARCH_COOLDOWN
                    and search_moves < MAX_SEARCH_MOVES):
                d, why = choose_direction(pipe, frame, arm, tried)
                if d is None:
                    note = f"search stopped: {why}"
                    search_moves = MAX_SEARCH_MOVES
                    print(f"  [SEARCH] giving up -- {why}")
                else:
                    dx, dy = SEARCH_DELTA[d]
                    try:
                        arm.update(dx, dy)
                    except Exception as e:
                        print(f"[ARM] search write FAILED: {e}")
                        reason = "arm write failure"; break
                    tried[d] += 1
                    search_moves += 1
                    cmds += 1
                    last_search_frame = frames
                    pipe.notify_self_motion()
                    ctrl.reset()
                    note = f"searching {d} ({search_moves}/{MAX_SEARCH_MOVES})"
                    print(f"  [SEARCH] {d}  -> ({arm.base},{arm.tilt})  "
                          f"{search_moves}/{MAX_SEARCH_MOVES}  [{why}]")

            if view:
                cv2.imshow("recovery pipeline",
                           draw(frame, pipe, err, move, arm, ctrl, fps,
                                target, note))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    reason = "quit (q)"; break
                if cv2.getWindowProperty("recovery pipeline",
                                         cv2.WND_PROP_VISIBLE) < 1:
                    reason = "window closed"; break
    except KeyboardInterrupt:
        reason = "interrupted (Ctrl+C)"
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

    el = time.time() - t0
    print("\n" + "=" * 64)
    print(f"target      {target!r}")
    print(f"stopped     {reason}")
    print(f"frames      {frames}   commands {cmds}   vlm {pipe.vlm_calls}")
    print(f"elapsed     {el:.1f}s" + (f"  ({frames / el:.1f} fps)" if el else ""))
    print(f"final       {pipe.status}    arm {(arm.base, arm.tilt)} "
          f"(started {start})")
    print("=" * 64)


if __name__ == "__main__":
    main()
