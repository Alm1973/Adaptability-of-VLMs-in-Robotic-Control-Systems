import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

import cv2

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

HOME = (90, 90)
DRIFT_TOLERANCE = 3
SETTLE_S = 1.0
PHASE_SECONDS = 6.0
EDGE_SECONDS = 1.0
TARGET = "red cup"

PASS_ACC = 0.70
PARTIAL_ACC = 0.40

OCCLUDER_OBJECT = "the cardboard box"
SUBSTITUTE_OBJECT = "the PINK PLIERS"
IMPOSTOR_OBJECT = ("a SECOND CUP OR MUG of a different colour (it does not "
                   "have to be the same model as the red cup)")

CLUTTER_LEVELS = {
    "none": ("BARE rig. Only the red cup on the marked centre position. "
             "Nothing else anywhere in frame."),
    "low": ("THREE other objects in frame, none touching the cup and none "
            "within a hand's width of it. Keep them the same three all "
            "session and write down what they are."),
    "high": ("SIX to EIGHT other objects in frame, including at least TWO "
             "other cups or mugs of different colours, crowded around the "
             "marked centre position without covering the red cup. Keep the "
             "same set all session and write down what they are."),
}

SCENARIOS = {
    "control": [
        ("baseline", True, "Cup at the MARKED CENTRE position, upright. "
                           "Do not touch anything."),
        ("disrupt", True, "Change NOTHING. Keep your hands out of frame."),
        ("recover", True, "Change NOTHING. Keep your hands out of frame."),
    ],
    "occlude_hand": [
        ("baseline", True, "Cup at the MARKED CENTRE position, upright. "
                           "Hands out of frame."),
        ("disrupt", True, "COVER the cup completely with your hand. "
                          "Hold it covered."),
        ("recover", True, "UNCOVER the cup. Leave it exactly where it is."),
    ],
    "occlude_object": [
        ("baseline", True, f"Cup at the MARKED CENTRE position, upright. "
                           f"{OCCLUDER_OBJECT} in your hand, out of frame."),
        ("disrupt", True, f"Stand {OCCLUDER_OBJECT} in front of the cup so the "
                          f"cup is COMPLETELY hidden. Take your hand away and "
                          f"keep it out of frame."),
        ("recover", True, f"REMOVE {OCCLUDER_OBJECT}. Leave the cup exactly "
                          f"where it is. Hands out."),
    ],
    "removal": [
        ("baseline", True, "Cup at the MARKED CENTRE position, upright. "
                           "Hands out of frame."),
        ("disrupt", True, "COVER the cup completely with your hand."),
        ("recover", False, "TAKE THE CUP AWAY while still covering it, then "
                           "remove your hand. The marked position must be "
                           "EMPTY and the cup must be OUT OF FRAME."),
    ],
    "substitution": [
        ("baseline", True, f"Cup at the MARKED CENTRE position, upright. "
                           f"{SUBSTITUTE_OBJECT} in your other hand, out of "
                           f"frame."),
        ("disrupt", True, "COVER the cup completely with your hand."),
        ("recover", False, f"SWAP: take the cup away and put "
                           f"{SUBSTITUTE_OBJECT} on the MARKED CENTRE "
                           f"position. The red cup must leave the frame "
                           f"ENTIRELY. Hands out."),
    ],
    "identity_swap": [
        ("baseline", True, "TARGET cup (red) at the MARKED CENTRE position, "
                           "upright. The IMPOSTOR -- any second cup or mug of "
                           "a DIFFERENT COLOUR, it does not have to match the "
                           "red cup's shape -- in your other hand, out of "
                           "frame."),
        ("disrupt", True, "COVER the red cup completely with your hand."),
        ("recover", False, "SWAP CUP FOR CUP in one motion: lift the RED cup "
                           "away and set the OTHER-COLOURED cup on the same "
                           "MARKED position. A cup must be visible before "
                           "and after -- if the position is ever EMPTY this "
                           "is a removal trial, not an identity trial. "
                           "Hands out."),
    ],

    "lighting": [
        ("baseline", True, "Cup at the MARKED CENTRE position, upright. Rig "
                           "lighting at its NORMAL level. Hands out of frame."),
        ("disrupt", True, "CHANGE THE LIGHTING -- switch the rig lamp off, or "
                          "dim it as far as it goes. Do NOT touch the cup and "
                          "do not move anything else."),
        ("recover", True, "RESTORE the lighting to its normal level. Still do "
                          "not touch the cup."),
    ],
    "camera_pose": [
        ("baseline", True, "Cup at the MARKED CENTRE position, upright. Hands "
                           "out of frame. THE ARM MOVES ITSELF in this trial "
                           "-- stay clear of it."),
        ("disrupt", True, "Hands clear. The arm pans itself. Do not touch "
                          "anything, including the cup."),
        ("recover", True, "Hands clear. The arm pans back. Do not touch "
                          "anything."),
    ],
}

ARM_MOVES = {
    "camera_pose": {"disrupt": (12, 0), "recover": (-12, 0)},
}

DEFAULT_SCENARIOS = ["control", "occlude_hand", "occlude_object", "removal",
                     "substitution", "identity_swap", "lighting",
                     "camera_pose"]

REQUIRES_PROP = {
    "occlude_object": OCCLUDER_OBJECT,
    "substitution": SUBSTITUTE_OBJECT,
    "identity_swap": IMPOSTOR_OBJECT,
}

AXIS = {
    "control": "control",
    "occlude_hand": "occlusion",
    "occlude_object": "occlusion",
    "removal": "occlusion",
    "substitution": "unexpected",
    "identity_swap": "identity",
    "lighting": "environment",
    "camera_pose": "environment",
}


def log(run_dir, msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(os.path.join(run_dir, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_order(scenarios, reps, seed, clutters=("none",)):
    rng = random.Random(seed)
    order = []
    for rep in range(1, reps + 1):
        levels = list(clutters)
        rng.shuffle(levels)
        for cl in levels:
            block = list(scenarios)
            rng.shuffle(block)
            order += [(name, rep, cl) for name in block]
    return order


def classify(acc, false_belief):
    if false_belief > 0:
        return "FAIL"
    if acc is None:
        return "FAIL"
    if acc >= PASS_ACC:
        return "PASS"
    if acc >= PARTIAL_ACC:
        return "PARTIAL"
    return "FAIL"


def home_arm(arm, run_dir, where):
    if arm is None:
        return True
    db, dt = arm.base - HOME[0], arm.tilt - HOME[1]
    if db or dt:
        arm.update(db, dt)
        time.sleep(SETTLE_S)
    drift = max(abs(arm.base - HOME[0]), abs(arm.tilt - HOME[1]))
    if drift > DRIFT_TOLERANCE:
        log(run_dir, f"ABORT {where}: arm failed to home, drift={drift}")
        return False
    return True


def annotate(frame, det, status, believes, truth, scenario, phase, t):
    a = frame.copy()
    h, w = a.shape[:2]
    if det:
        x, y, bw, bh = det["box"]
        cv2.rectangle(a, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        cv2.putText(a, f"{det.get('label','?')} {det.get('conf','')}",
                    (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0), 1)
    cv2.rectangle(a, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(a, f"{scenario} / {phase}  t={t:.1f}s", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    col = (0, 220, 0) if believes == truth else (0, 0, 235)
    cv2.putText(a, f"{status}  believes={believes}  truth="
                f"{'PRESENT' if truth else 'ABSENT'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    return a


_STDIN_Q = None


def _stdin_queue():
    global _STDIN_Q
    if _STDIN_Q is None:
        import queue
        import threading
        _STDIN_Q = queue.Queue()

        def reader():
            while True:
                try:
                    _STDIN_Q.put(input().strip().lower())
                except (EOFError, OSError):
                    _STDIN_Q.put(None)
                    return

        threading.Thread(target=reader, daemon=True).start()
    return _STDIN_Q


def _drain(q):
    import queue as _q
    while True:
        try:
            q.get_nowait()
        except _q.Empty:
            return


def prompt(msg, allowed):
    import queue as _q
    q = _stdin_queue()
    _drain(q)
    hint = "/".join(a if a else "ENTER" for a in allowed)
    print(f"  >>> {msg} [{hint}]: ", end="", flush=True)
    while True:
        try:
            got = q.get(timeout=0.2)
        except _q.Empty:
            continue
        if got is None:
            raise SystemExit("ABORT: no stdin. Run this in a terminal.")
        if got == "" and "" in allowed:
            return ""
        if got in allowed:
            return got
        print(f"      type one of: {hint}: ", end="", flush=True)


def prompt_live(cam, tracker, msg, allowed, header="", body=""):
    import queue as _q

    hint = "/".join(a if a else "ENTER" for a in allowed)
    q = _stdin_queue()
    _drain(q)
    print(f"  >>> {msg} [{hint}]  (feed is live)")

    while True:
        try:
            got = q.get_nowait()
            if got is None:
                raise SystemExit("ABORT: no stdin. Run this in a terminal.")
            if got == "" and "" in allowed:
                return ""
            if got in allowed:
                return got
            print(f"      type one of: {hint}")
        except _q.Empty:
            pass

        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        try:
            det = tracker.find(frame, TARGET)
        except Exception:
            det = None

        v = frame.copy()
        h, w = v.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.line(v, (cx, cy - 26), (cx, cy + 26), (255, 255, 255), 1)
        cv2.line(v, (cx - 26, cy), (cx + 26, cy), (255, 255, 255), 1)
        cv2.rectangle(v, (cx - 90, cy - 90), (cx + 90, cy + 90),
                      (120, 120, 120), 1)

        if det:
            x, y, bw, bh = det["box"]
            cv2.rectangle(v, (x, y), (x + bw, y + bh), (0, 230, 0), 3)
            dx, dy = det["center"][0] - cx, det["center"][1] - cy
            cv2.line(v, (cx, cy), det["center"], (0, 230, 0), 1)
            off = (dx * dx + dy * dy) ** 0.5
            state = ("CUP DETECTED  -- CENTRED" if off <= 90
                     else f"CUP DETECTED  -- off centre by {off:.0f}px")
            col = (0, 230, 0)
            sub = f"conf {det.get('conf','?')}   area {det['area']}   " \
                  f"dx={dx:+d} dy={dy:+d}"
        else:
            state, col = "NOT DETECTED", (0, 0, 255)
            sub = "the pipeline would call this a loss -- if you are covering " \
                  "the cup, this is what FULL occlusion looks like"

        band = 150 if (header or body) else 96
        cv2.rectangle(v, (0, 0), (w, band), (0, 0, 0), -1)
        cv2.putText(v, state, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.92, col, 2)
        cv2.putText(v, sub, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (210, 210, 210), 1)
        if header:
            cv2.putText(v, header, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 210, 255), 2)
        for i, line in enumerate(_wrap(body, 78)[:2]):
            cv2.putText(v, line, (12, 122 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        cv2.putText(v, f"answer in the TERMINAL  [{hint}]", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("AVI final experiment", v)
        cv2.waitKey(30)


def _wrap(text, width=60):
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def edge_window(phase_seconds):
    return min(EDGE_SECONDS, phase_seconds / 4.0)


def record_phase(cam, pipe, tracker, seconds, phase, present, scenario,
                 trial_dir, trial_no, run_dir, edge_s=None):
    if edge_s is None:
        edge_s = edge_window(seconds)
    recs = []
    t0 = time.time()
    stamp = datetime.now().strftime("%H%M%S")
    while time.time() - t0 < seconds:
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        dt = time.time() - t0

        t_inf0 = time.time()
        pipe.step(frame)
        inf_ms = (time.time() - t_inf0) * 1000.0

        try:
            det = tracker.find(frame, TARGET)
        except Exception:
            det = None

        edge = dt < edge_s or dt > seconds - edge_s
        base = (f"trial{trial_no:03d}_{scenario}_{phase}_"
                f"{len(recs):03d}_{stamp}")
        raw_p = os.path.join(trial_dir, base + "_raw.jpg")
        ann_p = os.path.join(trial_dir, base + "_annotated.jpg")
        cv2.imwrite(raw_p, frame)
        cv2.imwrite(ann_p, annotate(frame, det, pipe.status,
                                    pipe.believes_present(), present,
                                    scenario, phase, dt))

        recs.append({
            "phase": phase, "t": round(dt, 2), "present": present,
            "status": pipe.status, "believes": bool(pipe.believes_present()),
            "edge": edge,
            "raw": os.path.relpath(raw_p, run_dir),
            "annotated": os.path.relpath(ann_p, run_dir),
            "detected": bool(det),
            "detector_conf": (det or {}).get("conf"),
            "mean_brightness": round(float(frame.mean()), 1),
            "step_ms": round(inf_ms, 1),
            "vlm_calls": pipe.vlm_calls,
        })

        hud = annotate(frame, det, pipe.status, pipe.believes_present(),
                       present, scenario, phase, dt)
        cv2.putText(hud, f"{seconds - dt:0.0f}s",
                    (hud.shape[1] - 100, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 200, 255), 2)
        cv2.imshow("AVI final experiment", hud)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            raise KeyboardInterrupt("operator pressed q")
    return recs


def score_trial(recs):
    scored = [r for r in recs if not r["edge"]]
    dis = [r for r in scored if r["phase"] == "disrupt"]
    acc = (round(sum(r["believes"] == r["present"] for r in dis) / len(dis), 3)
           if dis else None)
    false_belief = sum(1 for r in scored if r["believes"] and not r["present"])
    lost = sum(1 for r in scored if r["present"] and not r["believes"])
    final_confirmed = bool(recs) and recs[-1]["status"] == "CONFIRMED"
    target_gone = bool(recs) and not recs[-1]["present"]
    return {
        "during_disruption_acc": acc,
        "false_belief_frames": false_belief,
        "lost_while_present": lost,
        "spurious_reacquisition": bool(final_confirmed and target_gone),
        "scored_frames": len(scored),
        "edge_frames": len(recs) - len(scored),
        "total_frames": len(recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5,
                    help="trials per scenario (default 5)")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--clutter", default="none,high",
                    help="comma-separated clutter levels to CROSS with every "
                         "scenario, e.g. none,high. Blocked, not interleaved "
                         "-- restaged once per block.")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--resume", default=None,
                    help="path to an existing runs/<stamp> folder")
    ap.add_argument("--smoke", action="store_true",
                    help="1 rep, 1.5s phases, no arm -- proves the harness "
                         "end to end without spending a real session")
    args = ap.parse_args()

    clutters = [c.strip() for c in args.clutter.split(",") if c.strip()]
    badc = [c for c in clutters if c not in CLUTTER_LEVELS]
    if badc:
        raise SystemExit(f"unknown clutter level(s) {badc}; "
                         f"choose from {list(CLUTTER_LEVELS)}")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    bad = [s for s in scenarios if s not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario(s): {bad}\n"
                         f"choose from: {list(SCENARIOS)}")

    if not args.resume:
        need = [(sc, REQUIRES_PROP[sc]) for sc in scenarios
                if sc in REQUIRES_PROP]
        if need:
            print("\nPROPS THIS RUN NEEDS -- have them in reach before you "
                  "start:")
            for sc, prop in need:
                print(f"  {sc:<16} {prop}")
            if prompt("all of these to hand?", ["y", "n"]) == "n":
                keep = []
                for sc in scenarios:
                    if sc in REQUIRES_PROP and prompt(
                            f"  have {REQUIRES_PROP[sc]}? (for {sc})",
                            ["y", "n"]) == "n":
                        print(f"    dropping {sc}")
                        continue
                    keep.append(sc)
                if "identity_swap" in scenarios and "identity_swap" not in keep:
                    print("\n  ** Dropping identity_swap means this run "
                          "cannot test the VLM's ONE claimed capability.")
                    print("     Finding 1 stays resting on synthetic data "
                          "alone, as it has since three failed attempts. **")
                if not keep:
                    raise SystemExit("ABORT: no scenarios left.")
                scenarios = keep

    phase_seconds = 1.5 if args.smoke else PHASE_SECONDS
    reps = 1 if args.smoke else args.reps

    if args.resume:
        run_dir = args.resume
        if not os.path.isdir(run_dir):
            raise SystemExit(f"no such run folder: {run_dir}")
        cfg = json.load(open(os.path.join(run_dir, "experiment_config.json")))
        order = [tuple(o) for o in cfg["order"]]
        if order and len(order[0]) != 3:
            raise SystemExit("ABORT: that run predates the clutter factor. "
                             "Resuming would mix two designs -- start a new "
                             "run instead.")
        clutters = cfg.get("clutter_levels", clutters)
        done = set()
        rp = os.path.join(run_dir, "results.json")
        rows = json.load(open(rp)) if os.path.exists(rp) else []
        for row in rows:
            done.add(row["trial_no"])
        print(f"RESUMING {run_dir}: {len(done)} trials already complete")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = os.path.join(_PROJECT_DIR, "runs", stamp)
        os.makedirs(os.path.join(run_dir, "baseline"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
        order = build_order(scenarios, reps, args.seed, clutters)
        rows, done = [], set()
        cfg = {
            "created": stamp,
            "hypothesis_primary": (
                "The VLM's contribution is confined to refusing false "
                "presence; it does not maintain belief through occlusion."),
            "independent_live": ["disruption", "clutter"],
            "independent_offline": ["ablation (D_full vs E_no_vlm)"],
            "dependent_primary": "false_belief_frames",
            "dependent_secondary": ["during_disruption_acc",
                                    "lost_while_present",
                                    "spurious_reacquisition"],
            "held_constant": ["camera pose (homed 90/90 each trial)",
                              "lighting", "target position", "occluder set",
                              "phase durations", "target string"],
            "success_criteria": {
                "PASS": f"acc >= {PASS_ACC} and false_belief == 0",
                "PARTIAL": f"acc >= {PARTIAL_ACC} and false_belief == 0",
                "FAIL": "otherwise, or any false belief at all",
            },
            "scenarios": scenarios, "reps": reps, "clutter_levels": clutters,
            "clutter_definitions": {c: CLUTTER_LEVELS[c] for c in clutters},
            "seed": args.seed, "phase_seconds": phase_seconds,
            "edge_seconds": round(edge_window(phase_seconds), 3),
            "target": TARGET,
            "smoke": bool(args.smoke),
            "order": order,
        }
        json.dump(cfg, open(os.path.join(run_dir, "experiment_config.json"),
                            "w"), indent=2)
        shutil.copyfile(__file__, os.path.join(run_dir,
                                               "final_experiment.py.snapshot"))

    log(run_dir, f"run dir: {run_dir}")
    log(run_dir, f"{len(order)} trials, clutter levels={clutters}, "
                 f"seed={args.seed}, smoke={args.smoke}")
    log(run_dir, f"phase={phase_seconds}s, edge margin="
                 f"{edge_window(phase_seconds):.2f}s each end -> "
                 f"{phase_seconds - 2*edge_window(phase_seconds):.2f}s "
                 f"scoreable per phase")

    from camera import Camera
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier
    from yolo_tracker import YoloTracker

    log(run_dir, "opening camera...")
    cam = Camera()
    tracker = YoloTracker(default_target=TARGET)
    tracker.set_targets([TARGET], allow_unreliable=True, quiet=True)
    log(run_dir, "loading verifier (this takes ~20s)...")
    verifier = make_fast_verifier()

    arm = None
    if not args.smoke:
        print("\nThe arm will home to 90/90 when the serial port opens.")
        if prompt("Arm clear and you are watching?", ["y", "n"]) == "n":
            raise SystemExit("ABORT: operator not ready.")
        from arm import Arm
        arm = Arm()
        time.sleep(SETTLE_S)
        if not home_arm(arm, run_dir, "startup"):
            raise SystemExit("ABORT: arm would not home.")
        log(run_dir, f"arm ready at base={arm.base} tilt={arm.tilt}")

    bdir = os.path.join(run_dir, "baseline")
    if not os.listdir(bdir):
        log(run_dir, "capturing baseline...")
        print("\nBASELINE: stage the rig BARE -- cup at the marked centre, "
              "NO clutter, hands out. This is the reference every later "
              "observation is diffed against, so it must be the clean scene.")
        prompt_live(cam, tracker, "ready?", ["", "y"],
                    header="BASELINE",
                    body="Cup on the marked centre, NO clutter, hands out. "
                         "Use the crosshair to centre it.")
        blines = []
        for i in range(10):
            ok, f = cam.read_fresh()
            if not ok or f is None:
                continue
            det = tracker.find(f, TARGET)
            p = os.path.join(bdir, f"baseline_{i:02d}_"
                                   f"{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(p, f)
            blines.append({"frame": os.path.relpath(p, run_dir),
                           "detected": bool(det),
                           "conf": (det or {}).get("conf"),
                           "mean_brightness": round(float(f.mean()), 1)})
        hit = sum(1 for b in blines if b["detected"])
        bright = [b["mean_brightness"] for b in blines]
        json.dump(blines, open(os.path.join(bdir, "baseline.json"), "w"),
                  indent=2)
        log(run_dir, f"baseline: cup detected in {hit}/{len(blines)} frames, "
                     f"brightness {min(bright):.0f}-{max(bright):.0f}")
        if hit < len(blines) * 0.8:
            print("\n  ** The cup is not reliably detected in the baseline.")
            print("     Every trial compares against this. Fix framing, "
                  "lighting or distance before continuing. **")
            if prompt("continue anyway?", ["y", "n"]) == "n":
                raise SystemExit("STOPPED at baseline, as instructed.")

    t_start = time.time()
    try:
        current_clutter = None
        for idx, (scenario, rep, clutter) in enumerate(order, start=1):
            if idx in done:
                continue
            phases = SCENARIOS[scenario]
            el = time.time() - t_start
            print("\n" + "=" * 72)
            print(f"TRIAL {idx} of {len(order)}   scenario={scenario}   "
                  f"rep={rep}   clutter={clutter}")
            print(f"elapsed {el/60:.1f} min")
            print("=" * 72)

            if clutter != current_clutter:
                print("")
                print("  *** CHANGE THE CLUTTER ***")
                print(f"  {CLUTTER_LEVELS[clutter]}")
                prompt_live(cam, tracker,
                            "clutter restaged? ENTER when done", ["", "y"],
                            header=f"CHANGE CLUTTER -> {clutter.upper()}",
                            body=CLUTTER_LEVELS[clutter])
                current_clutter = clutter
                log(run_dir, f"clutter changed to {clutter!r}")

            act = prompt_live(cam, tracker,
                              "ENTER=run, s=skip, a=abort run",
                              ["", "s", "a"],
                              header=f"TRIAL {idx}/{len(order)}  {scenario}"
                                     f"  rep {rep}  clutter {clutter}",
                              body="Reset the scene to the starting state, "
                                   "then press ENTER.")
            if act == "a":
                log(run_dir, f"trial {idx}: operator aborted the run")
                break
            if act == "s":
                log(run_dir, f"trial {idx} ({scenario}): SKIPPED by operator")
                continue

            while True:
                if not home_arm(arm, run_dir, f"trial {idx} start"):
                    raise SystemExit("ABORT: arm would not home.")

                tdir = os.path.join(run_dir, "trials",
                                    f"trial_{idx:03d}_{scenario}")
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)
                os.makedirs(tdir, exist_ok=True)

                pipe = RecoveryPipeline(TARGET, tracker=tracker,
                                        verifier=verifier, use_detector=True,
                                        use_opencv=True, use_vlm=True,
                                        use_state=True)
                recs, t0 = [], time.time()
                failure = None
                try:
                    for pi, (phase, present, instruction) in enumerate(phases, 1):
                        print(f"\n  STEP {pi}/{len(phases)} -- {phase.upper()}"
                              f"   (truth: "
                              f"{'PRESENT' if present else 'ABSENT'})")
                        print(f"  {instruction}")
                        prompt_live(cam, tracker,
                                    "done? ENTER to start recording",
                                    ["", "y"],
                                    header=f"{scenario}  /  {phase.upper()}"
                                           f"   truth: "
                                           f"{'PRESENT' if present else 'ABSENT'}",
                                    body=instruction)

                        mv = ARM_MOVES.get(scenario, {}).get(phase)
                        if mv:
                            if arm is None:
                                print(f"    [skipped arm move {mv} -- smoke "
                                      f"mode has no arm]")
                            else:
                                print(f"    moving arm {mv} ...")
                                arm.update(*mv)
                                time.sleep(SETTLE_S)
                                pipe.notify_self_motion()

                        recs += record_phase(cam, pipe, tracker,
                                             phase_seconds, phase, present,
                                             scenario, tdir, idx, run_dir)
                except KeyboardInterrupt as e:
                    failure = f"operator abort: {e}"
                except Exception as e:
                    failure = f"{type(e).__name__}: {e}"

                home_arm(arm, run_dir, f"trial {idx} end")

                if failure:
                    log(run_dir, f"trial {idx} ({scenario}): ABORTED -- {failure}")
                    verdict, sc = "ABORTED", score_trial(recs)
                else:
                    sc = score_trial(recs)
                    verdict = classify(sc["during_disruption_acc"],
                                       sc["false_belief_frames"])

                print(f"\n  RESULT  {verdict}")
                print(f"    during-disruption accuracy : "
                      f"{sc['during_disruption_acc']}")
                print(f"    FALSE BELIEF frames        : "
                      f"{sc['false_belief_frames']}   <- primary")
                print(f"    lost while present         : {sc['lost_while_present']}")
                print(f"    frames  {sc['total_frames']} "
                      f"({sc['scored_frames']} scored, {sc['edge_frames']} edge)")
                print(f"    VLM calls                  : "
                      f"{recs[-1]['vlm_calls'] if recs else 0}")

                again = prompt_live(cam, tracker,
                                    "ENTER=accept, r=repeat (fumbled "
                                    "staging), i=mark invalid",
                                    ["", "r", "i"],
                                    header=f"RESULT: {verdict}",
                                    body=f"acc="
                                         f"{sc['during_disruption_acc']}  "
                                         f"false belief="
                                         f"{sc['false_belief_frames']}")
                if again == "r":
                    log(run_dir, f"trial {idx} ({scenario}): REPEAT requested")
                    continue
                if again == "i":
                    verdict = "INVALID"
                    log(run_dir, f"trial {idx} ({scenario}): marked INVALID")

                row = {"trial_no": idx, "scenario": scenario, "rep": rep,
                       "clutter": clutter, "verdict": verdict,
                       "wall_clock_s": round(time.time() - t0, 1),
                       "failure_mode": failure,
                       "vlm_calls": recs[-1]["vlm_calls"] if recs else 0,
                       "trial_dir": os.path.relpath(tdir, run_dir),
                       **sc}
                row["frames"] = recs
                rows.append(row)
                done.add(idx)
                write_results(run_dir, rows)
                log(run_dir, f"trial {idx} ({scenario}) rep {rep}: {verdict}  "
                             f"acc={sc['during_disruption_acc']} "
                             f"false={sc['false_belief_frames']}")
                break
    except KeyboardInterrupt:
        log(run_dir, "run interrupted by operator (Ctrl-C)")
    finally:
        home_arm(arm, run_dir, "shutdown")
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        write_results(run_dir, rows)
        write_readme(run_dir, cfg, rows)

    summarise(run_dir, rows)


def write_results(run_dir, rows):
    json.dump(rows, open(os.path.join(run_dir, "results.json"), "w"),
              indent=2, default=str)
    cols = ["trial_no", "scenario", "rep", "clutter", "verdict",
            "during_disruption_acc", "false_belief_frames",
            "lost_while_present", "spurious_reacquisition", "scored_frames",
            "edge_frames", "total_frames", "vlm_calls", "wall_clock_s",
            "failure_mode", "trial_dir", "first_frame", "last_frame"]
    with open(os.path.join(run_dir, "results.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            fr = r.get("frames") or []
            w.writerow({**r,
                        "first_frame": fr[0]["raw"] if fr else "",
                        "last_frame": fr[-1]["raw"] if fr else ""})


def write_readme(run_dir, cfg, rows):
    p = os.path.join(run_dir, "README.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"""# AVI final experiment run — {cfg['created']}

## Hypothesis

{cfg['hypothesis_primary']}

## Variables

- **Independent (staged live):** {', '.join(cfg['independent_live'])}
- **Independent (applied offline):** {', '.join(cfg['independent_offline'])}
- **Dependent (primary):** `{cfg['dependent_primary']}` — asserting the target
  is present when it is not. This is the safety-relevant direction and is
  never folded into accuracy.
- **Dependent (secondary):** {', '.join(cfg['dependent_secondary'])}
- **Held constant:** {', '.join(cfg['held_constant'])}

## Success criteria (pre-registered, in code before the run)

| verdict | rule |
|---|---|
| PASS | {cfg['success_criteria']['PASS']} |
| PARTIAL | {cfg['success_criteria']['PARTIAL']} |
| FAIL | {cfg['success_criteria']['FAIL']} |

Any false belief is a FAIL regardless of accuracy.

## Conditions

{chr(10).join(f'- `{s}`' for s in cfg['scenarios'])}

Clutter levels this run: **{', '.join(cfg.get('clutter_levels', ['?']))}**.
{cfg['reps']} repetitions per scenario, order counterbalanced by block
(seed {cfg['seed']}).

## Folder structure

```
baseline/      scan captured before the trials, with baseline.json
trials/trial_<NNN>_<scenario>/
               every frame, as <trial>_<scenario>_<phase>_<idx>_<time>
               _raw.jpg  and  _annotated.jpg
results.csv    one row per trial
results.json   same, plus the full per-frame belief trajectory
run_log.txt    timestamped event log
experiment_config.json   the config this run was launched with
final_experiment.py.snapshot   the exact script that produced it
```

Raw frames are never overwritten. Each run gets its own folder.

## Ground truth

Recorded at capture time, never assigned by reviewing footage afterwards.
Frames within {cfg['edge_seconds']}s of a phase boundary are marked `edge`,
stepped through the pipeline so the belief trajectory stays continuous, and
excluded from grading.

## Trials completed

{len(rows)} rows in results.csv at the time this file was written.
""")


def summarise(run_dir, rows):
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    valid = [r for r in rows if r["verdict"] not in ("INVALID", "ABORTED")]
    if not valid:
        print("  no valid trials")
    else:
        names = sorted({r["scenario"] for r in valid})
        print(f"{'scenario':<18}{'n':>3}{'PASS':>6}{'PART':>6}{'FAIL':>6}"
              f"{'mean acc':>10}{'false belief':>14}")
        print("-" * 72)
        for s in names:
            sel = [r for r in valid if r["scenario"] == s]
            accs = [r["during_disruption_acc"] for r in sel
                    if r["during_disruption_acc"] is not None]
            print(f"{s:<18}{len(sel):>3}"
                  f"{sum(r['verdict']=='PASS' for r in sel):>6}"
                  f"{sum(r['verdict']=='PARTIAL' for r in sel):>6}"
                  f"{sum(r['verdict']=='FAIL' for r in sel):>6}"
                  f"{(sum(accs)/len(accs) if accs else 0):>10.3f}"
                  f"{sum(r['false_belief_frames'] for r in sel):>14}")
    inv = [r for r in rows if r["verdict"] in ("INVALID", "ABORTED")]
    if inv:
        print(f"\n  excluded: {len(inv)} "
              f"({sum(r['verdict']=='INVALID' for r in inv)} invalid, "
              f"{sum(r['verdict']=='ABORTED' for r in inv)} aborted)")
    print(f"\n  raw data: {run_dir}")
    print(f"    results.csv / results.json / run_log.txt / README.md")
    print("\n  Offline next: score every ablation (D_full vs E_no_vlm, etc.)")
    print("  on these same frames -- capture once, compare conditions on")
    print("  identical input.")


if __name__ == "__main__":
    main()
