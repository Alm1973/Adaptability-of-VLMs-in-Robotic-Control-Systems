import argparse
import time

import cv2

import findobject_search as fs
import llm_backend
from adaptive_detector import (LOCATE_PROMPT, Suppression, _ask, _parse_cell,
                               configure_validated, detect_candidates, verify)
from arm import Arm
from camera import Camera
from findobject_search import HOME, MOVE_PAUSE, Quit

REJECT_RETUNE = 3

DEADZONE_PX = 50
NUDGE_MAX_DEG = 7
NUDGE_GAIN_DIV = 55
NUDGE_COOLDOWN = 0.18
POST_NUDGE_SKIP = 1
LOST_TIMEOUT = 3.0
REVERIFY_SEC = 12.0

_VLM_CALLS = {"n": 0}
_orig_chat = llm_backend.vision_chat


def _counted_chat(*a, **k):
    _VLM_CALLS["n"] += 1
    return _orig_chat(*a, **k)


llm_backend.vision_chat = _counted_chat


JUMP_RADIUS_PX = 260


def _nearest_candidate(cands, pt):
    best, bd = None, 1e18
    for c in cands:
        cx, cy = c["center"]
        d = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
        if d < bd:
            best, bd = c, d
    return best, bd


def track_follow(arm, cam, cfg, query, supp, max_sec=0, start_center=None):
    sign_x, sign_y = 1, 1
    track_pt = start_center

    def _measure_center(tries=6):
        for _ in range(tries):
            okf, f = cam.read_fresh()
            if not okf or f is None:
                continue
            cs = detect_candidates(f, cfg, suppression=supp)
            if cs:
                c, d = _nearest_candidate(cs, track_pt) if track_pt \
                    else (cs[0], 0)
                if c is not None and d <= JUMP_RADIUS_PX:
                    return c["center"]
        return None

    CAL_DEG = 3
    calibrated = {"x": False, "y": False}
    for axis, (cdb, cdt) in (("x", (CAL_DEG, 0)), ("y", (0, CAL_DEG))):
        p0 = _measure_center()
        if p0 is None:
            break
        try:
            arm.update(cdb, cdt)
            time.sleep(0.35)
        except Exception:
            break
        for _ in range(2):
            cam.read_fresh()
        p1 = _measure_center()
        try:
            arm.update(-cdb, -cdt)
            time.sleep(0.35)
        except Exception:
            pass
        for _ in range(2):
            cam.read_fresh()
        if p1 is None:
            continue
        delta = (p1[0] - p0[0]) if axis == "x" else (p1[1] - p0[1])
        if abs(delta) >= 10:
            s = -1 if delta > 0 else 1
            if axis == "x":
                sign_x = s
            else:
                sign_y = s
            calibrated[axis] = True
            print(f"  axis {axis} calibrated: +{CAL_DEG}deg moved image "
                  f"{delta:+.0f}px -> sign {s:+d} (permanent)")

    last_nudge = 0.0
    last_err = {"x": None, "y": None}
    grew = {"x": 0, "y": 0}
    skip = 0
    last_seen = time.time()
    last_verify = time.time()
    n, ms_total, nudges = 0, 0.0, 0
    t_start = time.time()

    print(f"\n  tracking {query!r} -- servos follow; q/ESC or Ctrl+C stops")
    while True:
        if max_sec and time.time() - t_start > max_sec:
            return "timeout"
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        if skip > 0:
            skip -= 1
            continue

        t0 = time.time()
        cands = detect_candidates(frame, cfg, suppression=supp)
        dt_ms = (time.time() - t0) * 1000
        n += 1
        ms_total += dt_ms

        now = time.time()
        top = None
        if cands:
            if track_pt is None:
                top = cands[0]
                track_pt = top["center"]
            else:
                near, dist = _nearest_candidate(cands, track_pt)
                if dist <= JUMP_RADIUS_PX:
                    top = near
                    track_pt = top["center"]
                else:
                    ok_v, _ = verify(frame, near["box"], query)
                    last_verify = now
                    if ok_v:
                        print(f"  target re-acquired at "
                              f"{near['center']} (jump {dist:.0f}px, "
                              f"verified)")
                        top = near
                        track_pt = top["center"]
                    else:
                        supp.add(near["center"])
        hud = [f"TRACKING: {query}  base={arm.base} tilt={arm.tilt}",
               f"opencv {dt_ms:5.1f}ms  nudges={nudges}  "
               f"VLM calls={_VLM_CALLS['n']}"]

        if top:
            last_seen = now
            x, y, w, h = top["box"]
            cx, cy = x + w // 2, y + h // 2
            fh, fw = frame.shape[:2]
            err_x, err_y = cx - fw // 2, cy - fh // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            hud.append(f"err=({err_x:+d},{err_y:+d})px "
                       f"deadzone={DEADZONE_PX}")

            if now - last_verify > REVERIFY_SEC:
                last_verify = now
                ok_v, _ = verify(frame, top["box"], query)
                if ok_v is False:
                    print("  re-verify says this is NOT the target -> "
                          "suppressing and resuming search")
                    supp.add((cx, cy))
                    return "rejected"

            if (now - last_nudge > NUDGE_COOLDOWN and
                    (abs(err_x) > DEADZONE_PX or abs(err_y) > DEADZONE_PX)):
                db = dtl = 0
                if abs(err_x) > DEADZONE_PX:
                    db = sign_x * max(1, min(NUDGE_MAX_DEG,
                                             abs(err_x) // NUDGE_GAIN_DIV))
                    db = db if err_x > 0 else -db
                if abs(err_y) > DEADZONE_PX:
                    dtl = sign_y * max(1, min(NUDGE_MAX_DEG,
                                              abs(err_y) // NUDGE_GAIN_DIV))
                    dtl = dtl if err_y > 0 else -dtl
                nb = max(fs.BOUND_LO, min(fs.BOUND_HI, arm.base + db))
                nt = max(fs.BOUND_LO, min(fs.BOUND_HI, arm.tilt + dtl))
                db, dtl = nb - arm.base, nt - arm.tilt
                if db or dtl:
                    try:
                        arm.update(db, dtl)
                    except Exception as e:
                        print(f"  ARM_WRITE_FAILED during tracking: {e}")
                        return "lost"
                    if not arm.connection_healthy:
                        print("  ARM_UNHEALTHY during tracking")
                        return "lost"
                    nudges += 1
                    last_nudge = now
                    skip = POST_NUDGE_SKIP
                    for axis, err in (("x", err_x), ("y", err_y)):
                        if calibrated.get(axis):
                            continue
                        prev = last_err[axis]
                        if prev is not None and abs(err) > abs(prev) + 15:
                            grew[axis] += 1
                            if grew[axis] >= 2:
                                if axis == "x":
                                    sign_x = -sign_x
                                else:
                                    sign_y = -sign_y
                                grew[axis] = 0
                                print(f"  servo direction on {axis} was "
                                      f"backwards -> flipped")
                        else:
                            grew[axis] = 0
                    last_err["x"], last_err["y"] = err_x, err_y
        else:
            hud.append(f"no candidate ({now - last_seen:.1f}s / "
                       f"{LOST_TIMEOUT:.0f}s to lost)")
            if now - last_seen > LOST_TIMEOUT:
                print(f"  lost {query!r} ({LOST_TIMEOUT:.0f}s without a "
                      f"candidate) -> resuming search "
                      f"[{n} frames, mean {ms_total/max(1,n):.1f}ms, "
                      f"{nudges} nudges]")
                return "lost"

        if not fs.show(frame, hud):
            raise Quit()


SWEEP_STEP_DEG = 2
SWEEP_STEP_SLEEP = 0.09
LOCATE_EVERY_DEG = 15
STUCK_WINDOW = 6
STUCK_DIFF = 2.0

WAYPOINTS = [(45, 70), (135, 70), (135, 90), (45, 90), (45, 110), (135, 110)]


def search_continuous(arm, cam, query, passes, cfg=None, supp=None):
    supp = supp or Suppression()
    rejects = 0
    opencv_ms = []
    win_ref = None
    steps = 0

    def micro_move(db, dt):
        try:
            arm.update(db, dt)
        except Exception as e:
            return f"ARM_WRITE_FAILED: {e}"
        if not arm.connection_healthy:
            return "ARM_UNHEALTHY"
        return None

    for p in range(1, passes + 1):
        print(f"--- pass {p}/{passes} (continuous) ---")
        pass_cands = 0
        cfg_at_pass_start = cfg is not None
        since_locate = LOCATE_EVERY_DEG
        for wi, (wb, wt) in enumerate(WAYPOINTS):
            while arm.base != wb or arm.tilt != wt:
                db = max(-SWEEP_STEP_DEG, min(SWEEP_STEP_DEG, wb - arm.base))
                dt = max(-SWEEP_STEP_DEG, min(SWEEP_STEP_DEG, wt - arm.tilt))
                abort = micro_move(db, dt)
                if abort:
                    print(f"  ABORT: {abort}")
                    return None, abort, opencv_ms
                time.sleep(SWEEP_STEP_SLEEP)
                okf, frame = cam.read_fresh(drain=1)
                if not okf or frame is None:
                    continue
                steps += 1
                since_locate += abs(db) + abs(dt)

                if steps % STUCK_WINDOW == 0:
                    if win_ref is not None and \
                            fs.frame_diff(win_ref, frame) < STUCK_DIFF:
                        print(f"  ABORT: STUCK_ARM (no image change over "
                              f"{STUCK_WINDOW} steps)")
                        return None, "STUCK_ARM", opencv_ms
                    win_ref = frame

                hud = [f"searching: {query}",
                       f"pass {p}/{passes}  base={arm.base} tilt={arm.tilt}",
                       f"mode: {'OpenCV glide (' + '+'.join(cfg.cues) + ')' if cfg else 'glide + VLM checkpoints'}"]
                if not fs.show(frame, hud):
                    raise Quit()

                if cfg is not None:
                    t0 = time.time()
                    cands = detect_candidates(frame, cfg, suppression=supp)
                    opencv_ms.append((time.time() - t0) * 1000)
                    if not cands:
                        continue
                    pass_cands += len(cands)
                    frame2 = fs.settle(cam, hud + ["candidate: settling..."])
                    cands2 = detect_candidates(frame2, cfg, suppression=supp)
                    target = cands2[0] if cands2 else cands[0]
                    chk = frame2 if cands2 else frame
                    if target["area"] < 250:
                        continue
                    ok_v, vdt = verify(chk, target["box"], query)
                    if ok_v:
                        b, t = arm.base, arm.tilt
                        print(f"\n  ✅ FOUND {query!r} at base={b} tilt={t} "
                              f"box={target['box']} (verify {vdt:.2f}s)")
                        fs.show(chk, [f"FOUND: {query} - tracking",
                                      f"base={b} tilt={t}"], (0, 255, 0))
                        cv2.waitKey(600)
                        return (b, t, target["box"], cfg, supp), None, \
                            opencv_ms
                    supp.add(target["center"])
                    rejects += 1
                    print(f"  base={arm.base} tilt={arm.tilt}: candidate "
                          f"rejected ({rejects}/{REJECT_RETUNE} before "
                          f"retune)")
                    if rejects >= REJECT_RETUNE:
                        print("      too many rejections -> discarding "
                              "config to re-tune")
                        cfg, rejects = None, 0
                elif since_locate >= LOCATE_EVERY_DEG:
                    since_locate = 0
                    chk = fs.settle(cam, hud + ["checkpoint: looking..."])
                    if chk is None:
                        continue
                    cell = _parse_cell(_ask(LOCATE_PROMPT.format(obj=query),
                                            chk))
                    if cell is None:
                        continue
                    cell2 = _parse_cell(_ask(LOCATE_PROMPT.format(obj=query),
                                             chk))
                    if cell2 is None:
                        print(f"  base={arm.base} tilt={arm.tilt}  locate: "
                              f"{cell} unconfirmed on re-ask -> keep gliding")
                        continue
                    print(f"  base={arm.base} tilt={arm.tilt}  locate: "
                          f"{cell} (confirmed) -> configuring...")
                    cfg, cdt = configure_validated(chk, query, verbose=False,
                                                   allow_fallback=False)
                    if cfg is None:
                        print(f"      config not validated here "
                              f"({cdt:.1f}s) -> keep gliding")
                        continue
                    print(f"      configured in {cdt:.1f}s  cues={cfg.cues} "
                          f"colour={cfg.color_name!r}")
        if cfg is not None and cfg_at_pass_start and pass_cands == 0:
            print(f"  pass {p}: config produced ZERO candidates -> blind "
                  f"config; discarding to re-tune\n")
            cfg, rejects = None, 0
        else:
            print(f"  pass {p} complete\n")
    return None, None, opencv_ms


def search(arm, cam, query, passes, cfg=None, supp=None):
    poses = fs.build_poses()
    rejects = 0
    supp = supp or Suppression()
    opencv_ms = []
    prev = None
    print(f"\ngrid: {len(poses)} poses, up to {passes} passes\n")

    for p in range(1, passes + 1):
        print(f"--- pass {p}/{passes} ---")
        pass_cands = 0
        cfg_at_pass_start = cfg is not None
        for i, (b, t) in enumerate(poses, 1):
            status = [f"searching: {query}",
                      f"pass {p}/{passes}  pose {i}/{len(poses)}  "
                      f"base={b} tilt={t}",
                      f"mode: {'OpenCV (' + '+'.join(cfg.cues) + ')' if cfg else 'VLM locate'}"]
            frame, abort = fs.goto(arm, cam, b, t, prev, status)
            if abort:
                print(f"  ABORT: {abort}")
                return None, abort, opencv_ms
            prev = frame

            if cfg is None:
                cell = _parse_cell(_ask(LOCATE_PROMPT.format(obj=query),
                                        frame))
                if cell is None:
                    print(f"  [{i:2d}/{len(poses)}] b={b:3d} t={t:3d}  "
                          f"locate: not here")
                    continue
                print(f"  [{i:2d}/{len(poses)}] b={b:3d} t={t:3d}  "
                      f"locate: {cell} -> configuring...")
                cfg, cdt = configure_validated(frame, query, verbose=False,
                                               allow_fallback=False)
                if cfg is None:
                    print(f"      config not validated here ({cdt:.1f}s) -> "
                          f"keep sweeping (a bad config would blind the "
                          f"whole search)")
                    continue
                print(f"      configured in {cdt:.1f}s  cues={cfg.cues} "
                      f"colour={cfg.color_name!r}")

            t0 = time.time()
            cands = detect_candidates(frame, cfg, suppression=supp)
            dt_ms = (time.time() - t0) * 1000
            opencv_ms.append(dt_ms)
            if not cands:
                print(f"  [{i:2d}/{len(poses)}] b={b:3d} t={t:3d}  "
                      f"opencv {dt_ms:5.1f}ms: no candidates")
                continue
            pass_cands += len(cands)

            ok_v, vdt = verify(frame, cands[0]["box"], query)
            if ok_v:
                print(f"\n  ✅ FOUND {query!r} at base={b} tilt={t} "
                      f"box={cands[0]['box']} "
                      f"(opencv {dt_ms:.1f}ms + verify {vdt:.2f}s)")
                fs.show(frame, [f"FOUND: {query} - tracking",
                                f"base={b} tilt={t}"], (0, 255, 0))
                cv2.waitKey(800)
                return (b, t, cands[0]["box"], cfg, supp), None, opencv_ms
            supp.add(cands[0]["center"])
            rejects += 1
            print(f"  [{i:2d}/{len(poses)}] b={b:3d} t={t:3d}  "
                  f"opencv {dt_ms:5.1f}ms -> verify REJECTED "
                  f"({rejects}/{REJECT_RETUNE} before retune)")
            if rejects >= REJECT_RETUNE:
                print("      too many rejections -> config is tracking the "
                      "wrong thing; discarding it to re-tune (Role 1 again)")
                cfg, rejects = None, 0
        if cfg is not None and cfg_at_pass_start and pass_cands == 0:
            print(f"  pass {p}: config produced ZERO candidates all pass -> "
                  f"blind config; discarding to re-tune (Role 1 again)\n")
            cfg, rejects = None, 0
        else:
            print(f"  pass {p} complete\n")
    return None, None, opencv_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--track-sec", type=float, default=0,
                    help="stop tracking after N seconds (0 = forever); "
                         "used for bounded test runs")
    args = ap.parse_args()
    if args.no_window:
        fs._show_enabled = False

    query = " ".join(args.queries).strip()
    if not query:
        try:
            query = input("what should I search for? > ").strip()
        except EOFError:
            return
    if not query:
        return

    from query_fix import correct_query
    fixed, changed = correct_query(query)
    if changed:
        print(f' interpreting {query!r} as "{fixed}"')
        query = fixed

    print("=" * 62)
    print(" FINDSEARCH (final) -- VLM configures, OpenCV hunts, VLM verifies")
    print(" ⚠ MOVES THE ARM. q/ESC in window or Ctrl+C stops + homes.")
    print("=" * 62)
    print(f" target: {query!r}")

    cam = Camera()
    fs.settle(cam)
    print("\nopening arm (RESETS Arduino -> jumps to 90/90)...")
    arm = Arm()
    time.sleep(1.0)

    t0 = time.time()
    result = abort = None
    opencv_ms = []
    cfg = supp = None
    episodes = 0
    try:
        while True:
            result, abort, ms = search_continuous(arm, cam, query,
                                                  args.passes,
                                                  cfg=cfg, supp=supp)
            opencv_ms.extend(ms)
            if result is None:
                break
            episodes += 1
            b, t, box, cfg, supp = result
            outcome = track_follow(arm, cam, cfg, query, supp,
                                   max_sec=args.track_sec,
                                   start_center=(box[0] + box[2] // 2,
                                                 box[1] + box[3] // 2))
            if outcome == "timeout":
                break
            if outcome == "rejected":
                cfg = None
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Quit:
        print("\nstopped from live window")
    finally:
        print("\nreturning home...")
        try:
            arm.update(HOME[0] - arm.base, HOME[1] - arm.tilt)
            time.sleep(MOVE_PAUSE)
        except Exception as e:
            print(f"  (home failed: {e})")
        cam.release()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    wall = time.time() - t0
    print("\n===== SEARCH SUMMARY (placement: GPU) =====")
    if result:
        b, t, box, cfg, _ = result
        print(f"RESULT  : found {query!r} at base={b} tilt={t} box={box}")
        print(f"episodes: {episodes} find->track cycle(s)")
        print(f"config  : cues={cfg.cues} colour={cfg.color_name!r}")
    elif abort:
        print(f"RESULT  : aborted -- {abort}")
    else:
        print(f"RESULT  : not found after {args.passes} passes")
    print(f"wall    : {wall:.1f}s")
    print(f"VLM calls (exact, counted): {_VLM_CALLS['n']}")
    if opencv_ms:
        import statistics as st
        print(f"opencv  : {len(opencv_ms)} pose-checks, "
              f"mean {st.mean(opencv_ms):.1f} ms")
    old_est = 3 * (len(fs.build_poses()))
    print(f"old findsearch would have paid ~{old_est} VLM calls per pass; "
          f"this run paid {_VLM_CALLS['n']} total")


if __name__ == "__main__":
    main()
