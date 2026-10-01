import json
import os
import time

import cv2

from recovery_pipeline import (RecoveryPipeline, CONFIRMED, OCCLUDED,
                               DISPLACED, MISSING, AMBIGUOUS)

OUTDIR = "live_disruption_frames"
RESULTS = "live_disruption.json"
PHASE_SECONDS = 6.0
SETUP_SECONDS = 9.0
EDGE_SECONDS = 1.0
ARM_STEP = 10
BOUND_MARGIN = 25

SCENARIOS = [
    ("occlusion_full", [
        ("baseline", True, "Cup in clear view. Do NOT touch it."),
        ("disrupt", True, "COVER the cup completely with your hand. Keep it "
                          "covered."),
        ("recover", True, "UNCOVER the cup. Leave it where it is."),
    ], False),
    ("occlusion_remove", [
        ("baseline", True, "Cup in clear view. Do NOT touch it."),
        ("disrupt", True, "COVER the cup completely with your hand."),
        ("recover", False, "TAKE THE CUP AWAY while still covering, then "
                           "remove your hand. Desk must be EMPTY."),
    ], False),
    ("impostor", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "COVER the cup with your hand."),
        ("recover", False, "Put a DIFFERENT object where the cup was. The CUP "
                           "ITSELF MUST LEAVE THE FRAME COMPLETELY -- put it "
                           "on the floor or behind you, NOT elsewhere on the "
                           "desk. Then uncover."),
    ], False),
    ("lighting", [
        ("baseline", True, "Cup in clear view, normal lighting."),
        ("disrupt", True, "CHANGE THE LIGHT -- switch a lamp off, or shade "
                          "the desk. Do NOT touch the cup."),
        ("recover", True, "Restore the original lighting."),
    ], False),
    ("distractor", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "PLACE another object NEXT TO the cup. Leave the "
                          "cup visible."),
        ("recover", True, "Remove the other object."),
    ], False),
    ("occlude_book", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "Cover the cup with a BOOK or a box -- not your "
                          "hand. Keep your hand out of shot if you can."),
        ("recover", True, "Remove the book. Leave the cup where it is."),
    ], False),
    ("occlude_paper", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "Cover the cup with a SHEET OF PAPER or cloth."),
        ("recover", True, "Remove the paper. Leave the cup where it is."),
    ], False),
    ("occlude_box", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "Cover the cup with a CARDBOARD BOX. Keep your hand "
                          "out of shot once it is placed."),
        ("recover", True, "Remove the box. Leave the cup where it is."),
    ], False),
    ("occlude_jacket", [
        ("baseline", True, "Cup in clear view."),
        ("disrupt", True, "Cover the cup with a JACKET SLEEVE or folded "
                          "clothing. Hand out of shot if you can."),
        ("recover", True, "Remove it. Leave the cup where it is."),
    ], False),
    ("identity_same_class", [
        ("baseline", True, "The RED cup in clear view."),
        ("disrupt", True, "COVER the cup with your hand."),
        ("recover", False, "Swap in a cup of a DIFFERENT COLOUR (blue, green "
                           "or white) -- same size and shape, same spot. The "
                           "RED cup must leave the frame. Then uncover."),
    ], False),

    ("impostor_colour", [
        ("baseline", True, "The RED cup in clear view."),
        ("disrupt", True, "COVER the cup with your hand."),
        ("recover", False, "Swap in a cup of a DIFFERENT COLOUR, same spot and "
                           "size. The RED cup must be OUT OF FRAME (floor or "
                           "pocket). Then uncover."),
    ], False),

    ("clutter_similar", [
        ("baseline", True, "The red cup alone in clear view."),
        ("disrupt", True, "SURROUND the red cup with 2-4 other cups/mugs of "
                          "other colours, close together, partly overlapping. "
                          "The RED cup stays visible."),
        ("recover", True, "Remove the other cups. Red cup stays."),
    ], False),

    ("same_colour_occ", [
        ("baseline", True, "Red cup in clear view."),
        ("disrupt", True, "Partly cover the cup with a RED or ORANGE object "
                          "(book, cloth, box) -- about half the cup hidden."),
        ("recover", True, "Remove it. Cup stays."),
    ], False),

    ("thin_occluder", [
        ("baseline", True, "Red cup in clear view."),
        ("disrupt", True, "Put something THIN and slatted in front of the cup "
                          "-- a fork, comb, whisk, wire rack or open blinds. "
                          "Cup visible THROUGH the gaps."),
        ("recover", True, "Remove it."),
    ], False),

    ("distance_far", [
        ("baseline", True, "Red cup close to the camera, clear view."),
        ("disrupt", True, "Move the cup AS FAR from the camera as the room "
                          "allows, still in frame and unobstructed."),
        ("recover", True, "Bring it back to the original spot."),
    ], False),

    ("container", [
        ("baseline", True, "Red cup in clear view on the desk."),
        ("disrupt", True, "Put the cup INSIDE a box, drawer or mug so only the "
                          "rim or a sliver shows."),
        ("recover", True, "Take it back out, same spot."),
    ], False),

    ("transparent", [
        ("baseline", True, "Red cup in clear view."),
        ("disrupt", True, "Put something CLEAR in front of the cup -- a glass, "
                          "a plastic bottle, a food container. Cup seen "
                          "THROUGH it."),
        ("recover", True, "Remove it."),
    ], False),

    ("dynamic_pass", [
        ("baseline", True, "Red cup in clear view, you out of shot."),
        ("disrupt", True, "WALK slowly across between the camera and the cup, "
                          "two or three times. Do NOT touch the cup."),
        ("recover", True, "Step out of shot. Cup untouched."),
    ], False),

    ("camera_pose", [
        ("baseline", True, "Cup in clear view. Hands off -- the ARM will "
                           "move on its own."),
        ("disrupt", True, "(arm pans; do nothing)"),
        ("recover", True, "(arm pans back; do nothing)"),
    ], True),
]


WIN = "LIVE DISRUPTION -- follow the on-screen instruction"


def _wrap(text, width=44):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def show_instruction(cam, text, seconds, scenario, phase):
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            return True
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"{scenario}  /  {phase.upper()}", (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        for i, line in enumerate(_wrap(text)):
            cv2.putText(frame, line, (14, 66 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, f"starts in {left:0.0f}", (w - 210, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(frame, "q = abort", (w - 210, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(WIN, frame)
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            return False


def record(cam, pipe, seconds, phase, present, scenario, frames_out):
    recs = []
    probed = None
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        pipe.step(frame)

        if probed is None and pipe.status == OCCLUDED:
            label, is_occ = pipe.probe_region(frame)
            if label is not None:
                probed = {"label": label, "classified_as_occluder": is_occ}
                print(f"    [PROBE] region holds {label!r} -> "
                      f"{'occluder (stay)' if is_occ else 'replacement (drop)'}")

        dt = time.time() - t0
        edge = dt < EDGE_SECONDS or dt > seconds - EDGE_SECONDS
        fn = os.path.join(frames_out,
                          f"{scenario}_{phase}_{len(recs):03d}.jpg")
        cv2.imwrite(fn, frame)
        recs.append({"phase": phase, "t": round(dt, 2), "present": present,
                     "status": pipe.status,
                     "believes": pipe.believes_present(),
                     "edge": edge, "frame": fn, "probe": probed})
        print(f"    {dt:4.1f}s  {pipe.status:<10} "
              f"believes={pipe.believes_present()}"
              f"{'  (edge, not scored)' if edge else ''}")

        hud = frame.copy()
        h, w = hud.shape[:2]
        col = {CONFIRMED: (0, 220, 0), OCCLUDED: (0, 200, 255),
               DISPLACED: (255, 170, 0), MISSING: (0, 0, 235),
               AMBIGUOUS: (200, 200, 200)}.get(pipe.status, (255, 255, 255))
        cv2.rectangle(hud, (0, 0), (w, 92), (0, 0, 0), -1)
        cv2.putText(hud, f"RECORDING {scenario} / {phase.upper()}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hud, f"{pipe.status}   truth: "
                    f"{'PRESENT' if present else 'ABSENT'}", (14, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(hud, f"{seconds - dt:0.0f}s", (w - 110, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.imshow(WIN, hud)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            raise KeyboardInterrupt("operator aborted")
    return recs


def main():
    import shutil
    import sys

    only = None
    argv = list(sys.argv[1:])
    if "--only" in argv:
        i = argv.index("--only")
        only = [s for s in argv[i + 1].split(",") if s] if i + 1 < len(argv) \
            else []
        del argv[i:i + 2]
        known = {n for n, _, _ in SCENARIOS}
        bad = [s for s in only if s not in known]
        if bad:
            raise SystemExit(f"ABORT: unknown scenario(s) {bad}. "
                             f"Known: {sorted(known)}")
    target = " ".join(argv).strip() or "red cup"

    from camera import Camera
    from yolo_tracker import YoloTracker
    from run_study import make_fast_verifier as make_verifier

    print("=" * 70)
    print(" LIVE DISRUPTION VALIDATION -- a human performs each disruption")
    print("=" * 70)
    print(f" target: {target!r}")
    print(" One scenario MOVES THE ARM. Stay with the robot throughout.\n")

    os.makedirs(OUTDIR, exist_ok=True)
    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_verifier()

    all_recs = {}
    if os.path.exists(RESULTS):
        all_recs = json.load(open(RESULTS))
        if "records" in all_recs and isinstance(all_recs.get("records"), dict):
            all_recs = all_recs["records"]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(RESULTS, f"{RESULTS}.bak-{stamp}")
        if os.path.isdir(OUTDIR):
            shutil.copytree(OUTDIR, f"{OUTDIR}.bak-{stamp}")
        print(f" backed up existing corpus -> *.bak-{stamp}")
        print(f" carrying over: {sorted(all_recs)}")

    arm = None
    try:
        for name, phases, moves_arm in SCENARIOS:
            if only is not None and name not in only:
                continue
            print("\n" + "=" * 70)
            print(f" SCENARIO: {name}")
            print("=" * 70)
            if moves_arm and arm is None:
                from arm import Arm
                print(" opening arm (resets the Uno -> centres at 90/90)...")
                arm = Arm()
                time.sleep(2.5)

            pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier)
            recs = []
            for phase, present, instruction in phases:
                print(f"\n  [{phase.upper()}] {instruction}")
                if moves_arm and phase != "baseline":
                    d = ARM_STEP if phase == "disrupt" else -ARM_STEP
                    nb = arm.base + d
                    if not (BOUND_MARGIN < nb < 180 - BOUND_MARGIN):
                        print("    skipping arm move -- would exceed bounds")
                    else:
                        arm.update(d, 0)
                        if not arm.connection_healthy:
                            raise RuntimeError("arm link unhealthy")
                        pipe.notify_self_motion()
                        time.sleep(0.8)
                else:
                    if not show_instruction(cam, instruction, SETUP_SECONDS,
                                            name, phase):
                        raise KeyboardInterrupt("operator aborted")
                recs += record(cam, pipe, PHASE_SECONDS, phase, present,
                               name, OUTDIR)
            all_recs[name] = recs
            json.dump(all_recs, open(RESULTS, "w"), indent=2)
    except KeyboardInterrupt:
        print("\ninterrupted")
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

    print("\n" + "=" * 70)
    print("LIVE vs SYNTHETIC")
    print("=" * 70)
    print(f"{'scenario':<18}{'phase':<10}{'frames':>8}{'correct':>9}"
          f"{'states seen':>28}")
    print("-" * 73)
    summary = {}
    for name, recs in all_recs.items():
        summary[name] = {}
        for phase in ("baseline", "disrupt", "recover"):
            sel = [r for r in recs if r["phase"] == phase and not r["edge"]]
            if not sel:
                continue
            ok = sum(r["believes"] == r["present"] for r in sel)
            states = {}
            for r in sel:
                states[r["status"]] = states.get(r["status"], 0) + 1
            summary[name][phase] = {"n": len(sel), "correct": ok,
                                    "acc": round(ok / len(sel), 3),
                                    "states": states}
            st = " ".join(f"{k}:{v}" for k, v in sorted(states.items()))
            print(f"{name:<18}{phase:<10}{len(sel):>8}{ok:>9}   {st}")

    print("\nKEY CHECKS (stated before the run, not chosen after):")
    checks = []

    def check(label, cond, detail):
        checks.append((label, cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label} -- {detail}")

    o = summary.get("occlusion_full", {}).get("disrupt")
    if o:
        held = o["states"].get(OCCLUDED, 0)
        check("occlusion holds belief", held > 0,
              f"OCCLUDED on {held}/{o['n']} covered frames "
              f"(states: {o['states']})")
    r = summary.get("occlusion_remove", {}).get("recover")
    if r:
        check("removal is detected", r["acc"] >= 0.5,
              f"correct on {r['correct']}/{r['n']} frames after the cup was "
              f"taken away")
    i = summary.get("impostor", {}).get("recover")
    if i:
        conf = i["states"].get(CONFIRMED, 0)
        check("impostor rejected", conf == 0,
              f"CONFIRMED on {conf}/{i['n']} frames showing the swapped "
              f"object (want 0)")
    c = summary.get("camera_pose", {}).get("disrupt")
    if c:
        check("self-motion tolerated", c["acc"] >= 0.5,
              f"correct on {c['correct']}/{c['n']} frames during the "
              f"commanded pan")

    print("\nREGION PROBE -- can the VLM separate 'covered' from 'replaced'?")
    probes = {}
    for name, recs in all_recs.items():
        for r in recs:
            if r.get("probe") and name not in probes:
                probes[name] = r["probe"]
    if not probes:
        print("  no probe fired (belief never reached OCCLUDED) -- "
              "inconclusive, not negative")
    else:
        for name, p in probes.items():
            print(f"  {name:<18} saw {p['label']!r} -> classified as "
                  f"{'OCCLUDER (hold belief)' if p['classified_as_occluder'] else 'REPLACEMENT (drop belief)'}")
        occluder_scenarios = ("occlusion_full", "occlude_book", "occlude_paper")
        tested = {k: v for k, v in probes.items() if k in occluder_scenarios}
        if len(tested) > 1:
            good = [k for k, v in tested.items()
                    if v["classified_as_occluder"]]
            bad = [k for k, v in tested.items()
                   if not v["classified_as_occluder"]]
            print(f"\n  OCCLUDER GENERALISATION: {len(good)}/{len(tested)} "
                  f"types read as occluders")
            if bad:
                for k in bad:
                    print(f"    !! {k}: saw {probes[k]['label']!r} -> "
                          f"classified as a REPLACEMENT (belief dropped)")
                print("    -> OCCLUDER_WORDS does not generalise past the "
                      "hand. Belief will be\n       abandoned on ordinary "
                      "desk occluders. Needs a classifier, not a\n       "
                      "longer keyword list.")
            else:
                print("    -> the probe generalises across occluder types on "
                      "this scene.")

        occ = probes.get("occlusion_full", {})
        imp = probes.get("impostor", {})
        if occ and imp:
            good = (occ.get("classified_as_occluder") is True
                    and imp.get("classified_as_occluder") is False)
            print(f"  [{'PASS' if good else 'FAIL'}] hand read as occluder AND "
                  f"swapped object read as replacement")
            if good:
                print("  -> the timeout CAN be replaced by this question. "
                      "Enable use_region_probe\n     and re-run the study to "
                      "measure the gain.")
            else:
                print("  -> the distinction does NOT survive real occluders. "
                      "Keep the decay\n     timeout, and report that a fixed "
                      "timeout is the best available answer\n     despite not "
                      "generalising -- an honest negative.")

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n{passed}/{len(checks)} checks passed.")
    if checks and passed == len(checks):
        print("Live behaviour matches the synthetic predictions -- the "
              "synthetic corpus\nis validated as a proxy for real "
              "disruptions.")
    elif checks:
        print("Live behaviour DIVERGES from the synthetic predictions. The "
              "synthetic\nnumbers need an asterisk, and the divergence is "
              "itself the finding: it\nsays which part of the disruption "
              "model is unrealistic.")
    json.dump({"records": all_recs, "summary": summary},
              open(RESULTS, "w"), indent=2)
    print(f"\nwrote {RESULTS}, frames in {OUTDIR}/")


if __name__ == "__main__":
    main()
