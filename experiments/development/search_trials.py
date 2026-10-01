import json
import os
import sys
import time

import cv2
import numpy as np

from find_live import (BOUND_MARGIN, DECAY_FRAMES, SEARCH_DELTA,
                       MAX_PER_DIRECTION, choose_direction, _would_exceed,
                       make_verifier)

OUT = "search_trials.json"
BRIGHTNESS = [1.0, 0.5, 0.25, 0.12, 0.06]
TRIALS_PER = 2
START_OFFSETS = [(-50, 0), (50, 0)]
TIMEOUT_S = 25.0
SETTLE_S = 0.9

import find_live as _fl
_fl.MAX_PER_DIRECTION = 6
CENTRED_PX = 90


def darken(img, f):
    if f >= 1.0:
        return img
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


WIN = "AVI search -- what the pipeline sees"

STATE_COLOUR = {"CONFIRMED": (80, 220, 80), "OCCLUDED": (60, 200, 240),
                "DISPLACED": (240, 180, 60), "MISSING": (70, 70, 240),
                "AMBIGUOUS": (200, 200, 200)}


def show(frame, pipe, info):
    vis = frame.copy()
    h, w = vis.shape[:2]
    col = STATE_COLOUR.get(pipe.status, (200, 200, 200))

    cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 26, 1)
    cv2.circle(vis, (w // 2, h // 2), CENTRED_PX, (255, 255, 255), 1)

    if pipe.box is not None and pipe.status in ("CONFIRMED", "OCCLUDED"):
        x1, y1, x2, y2 = [int(v) for v in pipe.box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(vis, (w // 2, h // 2), (cx, cy), col, 1)

    cv2.rectangle(vis, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.putText(vis, pipe.status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, col, 2)
    cv2.putText(vis, info, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (230, 230, 230), 1)
    cv2.imshow(WIN, vis)
    return cv2.waitKey(1) & 0xFF


def run_trial(cam, arm, pipe, ctrl, bright, offset, home, log_dir, tag):
    from recovery_pipeline import CONFIRMED

    pipe.reset()
    ctrl.reset()
    tried = {d: 0 for d in SEARCH_DELTA}

    def target_visible():
        for _ in range(3):
            ok, probe_f = cam.read()
            if ok and probe_f is not None:
                hit, _others = pipe._detect(darken(probe_f, bright))
                return hit is not None
        return None

    step_dx = 15 if offset[0] > 0 else -15
    search_valid, travelled = None, 0
    for _ in range(5):
        if not _would_exceed(arm, step_dx, 0):
            arm.update(step_dx, 0)
            travelled += abs(step_dx)
            time.sleep(0.9)
        else:
            break
        if not arm.connection_healthy:
            raise RuntimeError("arm link unhealthy during offset")
        vis = target_visible()
        if vis is False:
            search_valid = True
            break
        search_valid = False
    if search_valid is False:
        print(f"    [!] cup STILL VISIBLE after {travelled} deg -- this trial "
              f"tests centring, not search")
    else:
        print(f"    offset {travelled} deg, cup out of frame")

    t0 = time.time()
    moves = 0
    found_at = None
    centred_at = None
    frames = 0
    last_err = None

    while time.time() - t0 < TIMEOUT_S:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        frame = darken(frame, bright)
        frames += 1
        pipe.step(frame)

        el = time.time() - t0
        key = show(frame, pipe, f"{tag}  bright {bright:.2f}  "
                                f"{el:4.1f}s / {TIMEOUT_S:.0f}s  "
                                f"moves {moves}  arm ({arm.base},{arm.tilt})")
        if key in (27, ord('q')):
            print("    [operator aborted this trial]")
            break

        if pipe.status == CONFIRMED and pipe.box is not None:
            if found_at is None:
                found_at = time.time() - t0
            h, w = frame.shape[:2]
            cx = (pipe.box[0] + pipe.box[2]) / 2
            cy = (pipe.box[1] + pipe.box[3]) / 2
            ex, ey = cx - w / 2, cy - h / 2
            last_err = float((ex ** 2 + ey ** 2) ** 0.5)
            if last_err <= CENTRED_PX:
                centred_at = time.time() - t0
                break
            move = ctrl.compute(ex, ey)
            if move and any(move):
                if not _would_exceed(arm, *move):
                    arm.update(*move)
                    pipe.notify_self_motion()
                    if not arm.connection_healthy:
                        raise RuntimeError("arm link unhealthy while centring")
        elif pipe.status == "MISSING":
            d, why = choose_direction(pipe, frame, arm, tried)
            if d is None:
                break
            sdx, sdy = SEARCH_DELTA[d]
            tried[d] += 1
            arm.update(sdx, sdy)
            pipe.notify_self_motion()
            moves += 1
            if not arm.connection_healthy:
                raise RuntimeError("arm link unhealthy while searching")
            ctrl.reset()
            time.sleep(SETTLE_S)

    if log_dir:
        ok, frame = cam.read()
        if ok and frame is not None:
            cv2.imwrite(os.path.join(log_dir, f"{tag}.jpg"), darken(frame, bright))

    return {"brightness": bright, "offset": list(offset),
            "search_valid": search_valid,
            "found": found_at is not None,
            "time_to_find": round(found_at, 2) if found_at else None,
            "centred": centred_at is not None,
            "time_to_centre": round(centred_at, 2) if centred_at else None,
            "search_moves": moves, "frames": frames,
            "final_err_px": round(last_err, 1) if last_err is not None else None,
            "vlm_calls": pipe.vlm_calls,
            "final_status": pipe.status}


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "red cup"
    log_dir = "search_trial_frames"
    os.makedirs(log_dir, exist_ok=True)

    from arm import Arm
    from camera import Camera
    from controller import Controller
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    print("=" * 68)
    print(" SEARCH TRIALS -- THE ARM WILL MOVE ON ITS OWN")
    print(" Stay with the robot. Ctrl-C stops and re-homes.")
    print("=" * 68)
    print(f" target {target!r}   brightness {BRIGHTNESS}")
    print(f" {len(BRIGHTNESS) * TRIALS_PER} trials, {TIMEOUT_S:.0f}s cap each\n")

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_verifier()

    arm = Arm()
    time.sleep(2.5)
    home = (arm.base, arm.tilt)
    print(f"home pose {home}\n")

    pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                            decay_frames=DECAY_FRAMES, use_region_probe=True,
                            probe_verifier=getattr(verifier, "generate", None))
    ctrl = Controller()

    rows = []
    try:
        for bright in BRIGHTNESS:
            for i in range(TRIALS_PER):
                offset = START_OFFSETS[i % len(START_OFFSETS)]
                tag = f"b{int(bright * 100):03d}_t{i}"
                print(f"[{tag}] brightness {bright:.2f}  start offset {offset}")
                db, dt = arm.base - home[0], arm.tilt - home[1]
                if db or dt:
                    arm.update(db, dt)
                    time.sleep(1.4)
                    print(f"    re-homed to ({arm.base},{arm.tilt})")

                drift = max(abs(arm.base - home[0]), abs(arm.tilt - home[1]))
                if drift > 3:
                    raise RuntimeError(
                        f"re-home failed: arm at ({arm.base},{arm.tilt}), "
                        f"home is {home}, drift {drift} deg. Refusing to "
                        f"continue -- trials from different start poses are "
                        f"not comparable, and a drifting arm ends at a stop.")
                r = run_trial(cam, arm, pipe, ctrl, bright, offset, home,
                              log_dir, tag)
                rows.append(r)
                print(f"    found={r['found']}  t={r['time_to_find']}  "
                      f"moves={r['search_moves']}  centred={r['centred']}  "
                      f"err={r['final_err_px']}  vlm={r['vlm_calls']}")
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    except Exception as e:
        print(f"\nABORT: {e}")
    finally:
        try:
            db, dt = arm.base - home[0], arm.tilt - home[1]
            if db or dt:
                print(f"\nre-homing to {home} from ({arm.base},{arm.tilt}) ...")
                arm.update(db, dt)
                time.sleep(1.5)
                print(f"  arm now at ({arm.base},{arm.tilt})")
        except Exception as e:
            print(f"  (re-home failed: {e})")
        cam.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 68)
    print("SEARCH SUCCESS vs BRIGHTNESS")
    print("=" * 68)
    print(f"{'brightness':>11}{'found':>8}{'centred':>9}{'med time':>10}"
          f"{'med moves':>11}")
    print("-" * 68)
    summary = {}
    for b in BRIGHTNESS:
        sel = [r for r in rows if r["brightness"] == b]
        if not sel:
            continue
        f = sum(r["found"] for r in sel)
        c = sum(r["centred"] for r in sel)
        ts = [r["time_to_find"] for r in sel if r["time_to_find"] is not None]
        mv = [r["search_moves"] for r in sel]
        summary[str(b)] = {"n": len(sel), "found": f, "centred": c,
                           "median_time": round(float(np.median(ts)), 2) if ts else None,
                           "median_moves": float(np.median(mv)) if mv else None}
        print(f"{b:>11.2f}{f:>4}/{len(sel):<3}{c:>6}/{len(sel):<3}"
              f"{str(summary[str(b)]['median_time']):>10}"
              f"{str(summary[str(b)]['median_moves']):>11}")

    print("\n  Static-frame reference (degrade_bench): detection survived to")
    print("  0.08x brightness at 0.735 confidence, and NEVER broke. If search")
    print("  fails at a brightness where static detection held, the difficulty")
    print("  is in the SEARCH -- motion blur, oblique angle, distance, time")
    print("  limit -- not in recognising the cup.")

    json.dump({"trials": rows, "summary": summary}, open(OUT, "w"), indent=2,
              default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
