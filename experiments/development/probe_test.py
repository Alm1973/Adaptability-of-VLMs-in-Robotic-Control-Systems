import json
import os
import re
import sys
from collections import Counter

import cv2

FRAMES = "live_disruption_frames"
OUT = "probe_test.json"
SAMPLE_EVERY = 6
MAX_PER_CASE = 10

CASES = [
    ("occlusion_full", "disrupt", True, "hand covering the cup"),
    ("impostor", "recover", False, "different object in its place"),
    ("occlusion_remove", "recover", False, "cup removed, empty desk"),
]


def frames_for(scenario, phase):
    out = []
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/")
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_{phase}_(\d+)\.jpg$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(FRAMES, fn)))
    return [p for _, p in sorted(out)]


def main():
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_verifier
    from yolo_tracker import YoloTracker
    import disruption_bench as db

    target = " ".join(sys.argv[1:]).strip() or "red cup"

    boxes = {}
    for scenario, _, _, _ in CASES:
        base = frames_for(scenario, "baseline")
        if not base:
            continue
        img = cv2.imread(base[len(base) // 2])
        got = db._target_box(img, target)
        if got:
            boxes[scenario] = got[1]
            print(f"{scenario:<18} box {got[1]} (conf {got[0]:.2f})")
    if not boxes:
        raise SystemExit("ABORT: could not locate the target in any baseline")

    print("\nloading GENERATIVE verifier (the probe needs open-ended output)...")
    verifier = make_verifier(max_new_tokens=12)
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)

    rows = []
    for scenario, phase, want_occluder, desc in CASES:
        if scenario not in boxes:
            print(f"\n-- {scenario}/{phase}: no box, skipped")
            continue
        paths = frames_for(scenario, phase)[::SAMPLE_EVERY][:MAX_PER_CASE]
        if not paths:
            continue
        print(f"\n-- {scenario}/{phase}: {desc}  ({len(paths)} frames)")
        pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                                use_state=True, probe_verifier=verifier)
        pipe.box = boxes[scenario]
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            label, is_occ = pipe.probe_region(img)
            rows.append({"scenario": scenario, "phase": phase,
                         "want_occluder": want_occluder,
                         "label": label, "is_occluder": is_occ,
                         "correct": (is_occ == want_occluder)
                         if label is not None else None})
            print(f"     {str(label):<28} -> "
                  f"{'OCCLUDER' if is_occ else 'replacement'}"
                  f"{'  OK' if is_occ == want_occluder else '  WRONG'}")

    json.dump(rows, open(OUT, "w"), indent=2)

    print("\n" + "=" * 74)
    print("CAN THE VLM SEPARATE 'COVERED' FROM 'REPLACED'?")
    print("=" * 74)
    ok_all = True
    for scenario, phase, want, desc in CASES:
        sel = [r for r in rows if r["scenario"] == scenario
               and r["phase"] == phase and r["label"] is not None]
        if not sel:
            continue
        good = sum(r["correct"] for r in sel)
        words = Counter(r["label"] for r in sel).most_common(3)
        print(f"\n  {scenario}/{phase}  ({desc})")
        print(f"    want {'OCCLUDER' if want else 'replacement'}: "
              f"{good}/{len(sel)} correct")
        print(f"    said: {', '.join(f'{w!r} x{n}' for w, n in words)}")
        if good < len(sel) * 0.7:
            ok_all = False

    print("\n" + "-" * 74)
    if ok_all and rows:
        print("PASS -- the distinction survives real occluders. The "
              "OCCLUDED->MISSING\ntimeout can be replaced by this question: "
              "enable use_region_probe, then\nre-run the study to measure "
              "what it is worth.")
    else:
        print("FAIL -- the VLM does not reliably separate a covering hand "
              "from a\nreplacement on real frames. Keep the decay timeout and "
              "report that a\nnon-generalising fixed value is the best "
              "available answer. That is an\nhonest negative, and it bounds "
              "what semantics can contribute here.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
