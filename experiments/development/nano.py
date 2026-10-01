import json
import os
import sys
import time

import cv2

CONDITIONS = {
    "clear": (1.0, "Red cup in plain view on the desk, nothing near it. "
                   "CONTROL -- run first, and again whenever the desk "
                   "changes, so later failures can be blamed on the condition "
                   "rather than the room.", True),
    "dark_50": (0.5, "No physical change. Brightness halved in software.",
                False),
    "dark_25": (0.25, "No physical change. Quarter brightness in software.",
                False),
    "dark_12": (0.12, "No physical change. 0.12x brightness in software.",
                False),
    "dark_06": (0.06, "No physical change. 0.06x brightness -- expected to "
                      "fail; it is the floor of the range.", False),
    "partial": (1.0, "Put an object in front of the cup so ABOUT HALF is "
                     "hidden. Cup still clearly visible.", True),
    "mostly": (1.0, "Cover the cup so only about a QUARTER shows -- near the "
                    "85% limit where detection is known to break.", True),
    "box": (1.0, "Stand a cardboard BOX in front of the cup so it is fully "
                 "hidden from the home position. Hands out of shot.", True),
    "cloth": (1.0, "Drape a CLOTH or towel over the cup, fully covering it.",
              True),
    "thin": (1.0, "Something THIN and slatted in front -- fork, comb, whisk, "
                  "wire rack, open blinds. Cup visible THROUGH the gaps.",
             True),
    "transparent": (1.0, "Something CLEAR in front -- glass, plastic bottle, "
                         "food container. Cup seen THROUGH it.", True),
    "far": (1.0, "Move the cup AS FAR from the camera as the room allows, "
                 "still unobstructed. Attacks the one measured perceptual "
                 "limit: the detector dies below 0.12x apparent size.", True),
    "clutter": (1.0, "Surround the cup with 2-4 OTHER cups/mugs of different "
                     "colours, close and partly overlapping. Red cup stays "
                     "visible.", True),
    "impostor": (1.0, "Put a cup of a DIFFERENT COLOUR where the red cup was, "
                      "and take the RED cup COMPLETELY OUT OF THE ROOM.",
                 True),
    "shadow": (1.0, "Put the cup in a genuine SHADOW -- under a shelf, behind "
                    "the monitor -- while the room stays lit. Physical "
                    "shadow, not software dimming.", True),
    "glare": (1.0, "Aim a lamp or phone torch at the cup so it has a bright "
                   "specular highlight.", True),
}

DARK_SET = ["clear", "dark_50", "dark_25", "dark_12", "dark_06"]
EXPECT_ABSENT = {"impostor"}


def paste_block(cond, bright, rows):
    print("\n" + "=" * 74)
    print("PASTE-ME  (copy everything between the lines)")
    print("=" * 74)
    print(f"NANO {cond} bright={bright} n={len(rows)} "
          f"expect={'absent' if cond in EXPECT_ABSENT else 'present'}")
    for i, r in enumerate(rows):
        print(f"  t{i} valid={r.get('search_valid')} found={r['found']} "
              f"tfind={r['time_to_find']} moves={r['search_moves']} "
              f"centred={r['centred']} err={r['final_err_px']} "
              f"vlm={r['vlm_calls']} frames={r['frames']} "
              f"status={r['final_status']}")
    found = sum(bool(r["found"]) for r in rows)
    cent = sum(bool(r["centred"]) for r in rows)
    valid = sum(r.get("search_valid") is True for r in rows)
    print(f"  TOTAL found={found}/{len(rows)} centred={cent}/{len(rows)} "
          f"searchvalid={valid}/{len(rows)}")
    if rows and valid < len(rows):
        print("  NOTE: valid=False trials did NOT test search -- the cup was "
              "still in\n  frame after the offset, so only centring was "
              "exercised.")
    print("=" * 74)


def run_condition(cond, repeats, ctx):
    from search_trials import run_trial, START_OFFSETS
    cam, arm, pipe, ctrl, home = ctx
    bright, setup, _needs = CONDITIONS[cond]

    print("\n" + "=" * 74)
    print(f" NANO: {cond}     brightness {bright:.2f}     {repeats} trial(s)")
    print("=" * 74)
    print(f"   {setup}")
    if cond in EXPECT_ABSENT:
        print("\n   !! EXPECTED: found=False. A found=True here is a FALSE "
              "POSITIVE,\n      and it is the most important failure we can "
              "record.")

    rows = []
    for i in range(repeats):
        db, dt = arm.base - home[0], arm.tilt - home[1]
        if db or dt:
            arm.update(db, dt)
            time.sleep(1.4)
        drift = max(abs(arm.base - home[0]), abs(arm.tilt - home[1]))
        if drift > 3:
            raise RuntimeError(
                f"re-home failed: arm at ({arm.base},{arm.tilt}) vs home "
                f"{home}, drift {drift} deg -- aborting rather than recording "
                f"trials from a moving start pose")
        off = START_OFFSETS[i % len(START_OFFSETS)]
        print(f"\n trial {i + 1}/{repeats}  offset {off}")
        r = run_trial(cam, arm, pipe, ctrl, bright, off, home,
                      "nano_frames", f"{cond}_{i}")
        r["condition"] = cond
        rows.append(r)
        print(f"   valid={r.get('search_valid')} found={r['found']} "
              f"t={r['time_to_find']} moves={r['search_moves']} "
              f"centred={r['centred']} err={r['final_err_px']}")

    json.dump(rows, open(f"nano_{cond}.json", "w"), indent=2, default=str)
    paste_block(cond, bright, rows)
    return rows


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("list", "-h", "--help"):
        print("usage: nano.bat <condition|dark|all> [repeats]   "
              "(repeats default 3)\n")
        print(f"{'condition':<14}{'bright':>7}{'setup?':>8}  what to stage")
        print("-" * 78)
        for k, (b, d, needs) in CONDITIONS.items():
            first = d if len(d) <= 46 else d[:43] + "..."
            print(f"{k:<14}{b:>7.2f}{('yes' if needs else 'no'):>8}  {first}")
        print("\n  nano.bat dark   -> clear + all 4 dark levels, "
              "NO physical setup between them")
        print("  nano.bat all    -> every condition, prompts you between each")
        return 0

    mode = argv[0]
    repeats = int(argv[1]) if len(argv) > 1 else 3

    if mode == "dark":
        queue = list(DARK_SET)
    elif mode == "all":
        queue = list(CONDITIONS)
    elif mode in CONDITIONS:
        queue = [mode]
    else:
        raise SystemExit(f"unknown condition {mode!r}. `nano.bat list`.")

    print(f"queue: {', '.join(queue)}   ({repeats} trials each, "
          f"{len(queue) * repeats} total)")
    print("The arm will move on its own. Stay with the robot.")

    from arm import Arm
    from camera import Camera
    from controller import Controller
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker
    from find_live import DECAY_FRAMES, make_verifier

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")
    target = "red cup"
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier (once for the whole queue)...")
    verifier = make_verifier()
    arm = Arm()
    time.sleep(2.5)
    home = (arm.base, arm.tilt)
    pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                            decay_frames=DECAY_FRAMES, use_region_probe=True,
                            probe_verifier=getattr(verifier, "generate", None))
    ctrl = Controller()
    os.makedirs("nano_frames", exist_ok=True)
    ctx = (cam, arm, pipe, ctrl, home)

    all_rows = {}
    try:
        for n, cond in enumerate(queue):
            _b, setup, needs = CONDITIONS[cond]
            if needs and len(queue) > 1:
                print("\n" + "-" * 74)
                print(f" NEXT ({n + 1}/{len(queue)}): {cond}")
                print(f"   {setup}")
                try:
                    input("\n   set the scene, then press ENTER (Ctrl-C to "
                          "stop)... ")
                except EOFError:
                    print("   (no console input available -- continuing)")
            all_rows[cond] = run_condition(cond, repeats, ctx)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    except Exception as e:
        print(f"\nABORT: {e}")
    finally:
        try:
            db, dt = arm.base - home[0], arm.tilt - home[1]
            if db or dt:
                arm.update(db, dt)
                time.sleep(1.5)
            print(f"\narm re-homed to ({arm.base},{arm.tilt})")
        except Exception as e:
            print(f" (re-home failed: {e})")
        cam.release()
        cv2.destroyAllWindows()

    if len(all_rows) > 1:
        print("\n" + "=" * 74)
        print("ALL CONDITIONS -- paste this whole block")
        print("=" * 74)
        print(f"{'condition':<14}{'valid':>8}{'found':>8}{'centred':>9}")
        for cond, rows in all_rows.items():
            v = sum(r.get("search_valid") is True for r in rows)
            f = sum(bool(r["found"]) for r in rows)
            c = sum(bool(r["centred"]) for r in rows)
            print(f"{cond:<14}{v:>4}/{len(rows):<3}{f:>4}/{len(rows):<3}"
                  f"{c:>5}/{len(rows):<3}")
        print("=" * 74)
        json.dump(all_rows, open("nano_all.json", "w"), indent=2, default=str)
        print("(saved nano_all.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
