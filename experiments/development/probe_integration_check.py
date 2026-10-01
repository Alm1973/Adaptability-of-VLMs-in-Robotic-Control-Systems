import os
import re

import cv2

FRAMES = "live_disruption_frames"

OCC_CASES = [
    ("occlusion_full", "real hand"),
    ("occlude_book", "real book"),
    ("occlude_paper", "real paper/cloth"),
    ("occlude_box", "real cardboard box"),
    ("occlude_jacket", "real jacket sleeve"),
]
REP_CASES = [
    ("occlusion_remove", "recover", "cup genuinely removed"),
]


def frames_for(scenario, phase):
    out = []
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_{phase}_(\d+)\.jpg$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(FRAMES, fn)))
    return [p for _, p in sorted(out)]


def baseline_box(scenario, target):
    import disruption_bench as db
    for fn in sorted(os.listdir(FRAMES)):
        if re.match(rf"^{re.escape(scenario)}_baseline_(\d+)\.jpg$", fn):
            got = db._target_box(cv2.imread(os.path.join(FRAMES, fn)), target)
            if got:
                return got[1]
    return None


def main():
    target = "red cup"
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier
    from clip_classifier import make_clip_classifier

    v = make_fast_verifier()
    gen = getattr(v, "generate", None)
    if gen is None:
        raise SystemExit("ABORT: probe needs a GENERATIVE verifier")
    clf = make_clip_classifier()

    pipe = RecoveryPipeline(target, tracker=None, verifier=None,
                            use_detector=False, use_opencv=True,
                            use_vlm=False, use_state=True,
                            use_region_probe=True, probe_verifier=gen,
                            probe_classifier=clf)

    print(f"{'case':<20}{'want':>6}{'votes':>10}{'verdict':>10}")
    print("-" * 50)
    total_ok, total_n = 0, 0
    for scenario, desc in OCC_CASES:
        box = baseline_box(scenario, target)
        paths = frames_for(scenario, "disrupt")
        if box is None or not paths:
            print(f"{scenario:<20}(no box/frames)")
            continue
        pipe.box = box
        mid = paths[len(paths) // 3: len(paths) // 3 * 2] or paths
        picks = mid[::max(1, len(mid) // 5)][:5]
        votes = []
        for p in picks:
            img = cv2.imread(p)
            pipe._probe_at = None
            pipe._probe_verdict = None
            _, is_occ = pipe.probe_region(img)
            votes.append(bool(is_occ))
        ok = sum(votes)
        total_ok += ok; total_n += len(votes)
        print(f"{scenario:<20}{'OCC':>6}{f'{ok}/{len(votes)}':>10}"
              f"{('OK' if ok > len(votes)/2 else 'WRONG'):>10}")

    for scenario, phase, desc in REP_CASES:
        box = baseline_box(scenario, target)
        paths = frames_for(scenario, phase)
        if box is None or not paths:
            print(f"{scenario:<20}(no box/frames)")
            continue
        pipe.box = box
        mid = paths[len(paths) // 3: len(paths) // 3 * 2] or paths
        picks = mid[::max(1, len(mid) // 5)][:5]
        votes = []
        for p in picks:
            img = cv2.imread(p)
            pipe._probe_at = None
            pipe._probe_verdict = None
            _, is_occ = pipe.probe_region(img)
            votes.append(bool(is_occ))
        ok = sum(1 for v in votes if not v)
        total_ok += ok; total_n += len(votes)
        print(f"{scenario + '/' + phase:<20}{'rep':>6}{f'{ok}/{len(votes)}':>10}"
              f"{('OK' if ok > len(votes)/2 else 'WRONG'):>10}")

    print(f"\ntotal: {total_ok}/{total_n} scenario-majority-correct frames")


if __name__ == "__main__":
    main()
