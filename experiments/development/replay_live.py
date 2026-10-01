import json
import os
import re
import sys
from collections import Counter

import cv2

FRAMES = "live_disruption_frames"
OUT = "replay_live.json"
PRESENT_STATES = ("CONFIRMED", "OCCLUDED")


def load_groups():
    groups = {}
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/ -- run live_disruption.py first")
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(r"^(.*)_(baseline|disrupt|recover)_(\d+)\.jpg$", fn)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        groups.setdefault(key, []).append((int(m.group(3)),
                                           os.path.join(FRAMES, fn)))
    return {k: [p for _, p in sorted(v)] for k, v in groups.items()}


PHASE_ORDER = ("baseline", "disrupt", "recover")


def run_episode(pipe, phases):
    out = {}
    pipe.reset()
    for phase in PHASE_ORDER:
        states = []
        for p in phases.get(phase, []):
            img = cv2.imread(p)
            if img is None:
                continue
            pipe.step(img)
            states.append(pipe.status)
        if states:
            out[phase] = states
    return out


def main():
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    target = " ".join(sys.argv[1:]).strip() or "red cup"
    groups = load_groups()
    if not groups:
        raise SystemExit("ABORT: no frames matched the expected naming")

    scenarios = sorted({s for s, _ in groups})
    total = sum(len(v) for v in groups.values())
    print(f"{len(groups)} scenario/phase groups, {total} frames, "
          f"target={target!r}")
    print(f"scenarios: {', '.join(scenarios)}\n")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)

    def make(pc):
        return RecoveryPipeline(target, tracker=tracker, verifier=None,
                                use_detector=True, use_opencv=True,
                                use_vlm=False, use_state=True,
                                periphery_change=pc)

    by_scenario = {}
    for (scenario, phase), paths in groups.items():
        by_scenario.setdefault(scenario, {})[phase] = paths

    probe_v = None
    if "--probe" in sys.argv:
        from run_study import make_verifier
        print("loading generative verifier for the region probe...")
        probe_v = make_verifier(max_new_tokens=12)

    def make_probe():
        return RecoveryPipeline(target, tracker=tracker, verifier=None,
                                use_detector=True, use_opencv=True,
                                use_vlm=False, use_state=True,
                                use_region_probe=True, probe_verifier=probe_v)

    results = {}
    for scenario in sorted(by_scenario):
        phases = by_scenario[scenario]
        before = run_episode(make(0.0), phases)
        after = run_episode(make(None), phases)
        probe = run_episode(make_probe(), phases) if probe_v else {}
        for phase in PHASE_ORDER:
            if phase not in before:
                continue
            results[f"{scenario}/{phase}"] = {
                "n": len(before[phase]),
                "before": dict(Counter(before[phase])),
                "after": dict(Counter(after.get(phase, []))),
                "probe": dict(Counter(probe.get(phase, []))),
            }

    print("=" * 78)
    print("BELIEF STATES ON REAL FRAMES -- before vs after the periphery fix")
    print("=" * 78)
    for k, r in results.items():
        def fmt(c):
            return " ".join(f"{s}:{n}" for s, n in
                            sorted(c.items(), key=lambda x: -x[1]))
        print(f"\n{k}  (n={r['n']})")
        print(f"   before  {fmt(r['before'])}")
        print(f"   after   {fmt(r['after'])}")
        if r.get("probe"):
            print(f"   probe   {fmt(r['probe'])}")

    key = "occlusion_full/disrupt"
    print("\n" + "=" * 78)
    print("THE CLAIM: a real hand should read as OCCLUDED, not DISPLACED")
    print("=" * 78)
    if key not in results:
        print(f"  no {key} frames -- cannot check")
    else:
        b, a = results[key]["before"], results[key]["after"]
        n = results[key]["n"]
        print(f"  covered frames: {n}")
        print(f"  DISPLACED   {b.get('DISPLACED', 0):>4} -> "
              f"{a.get('DISPLACED', 0):>4}")
        print(f"  OCCLUDED    {b.get('OCCLUDED', 0):>4} -> "
              f"{a.get('OCCLUDED', 0):>4}")
        bel_b = sum(v for s, v in b.items() if s in PRESENT_STATES)
        bel_a = sum(v for s, v in a.items() if s in PRESENT_STATES)
        print(f"  believes present (GT: the cup IS there)  "
              f"{bel_b}/{n} -> {bel_a}/{n}")
        if a.get("OCCLUDED", 0) > b.get("OCCLUDED", 0) and bel_a > bel_b:
            print("\n  FIX CONFIRMED on real frames: the hand now reads as "
                  "occlusion and\n  belief is held through it, which is the "
                  "behaviour the belief state\n  machine exists to produce.")
        elif bel_a > bel_b:
            print("\n  Belief improved but not via OCCLUDED -- check which "
                  "state it landed\n  in before claiming the mechanism works.")
        else:
            print("\n  NOT FIXED. The hand still does not read as occlusion. "
                  "The periphery\n  test is not the whole story -- do not "
                  "report this as solved.")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
