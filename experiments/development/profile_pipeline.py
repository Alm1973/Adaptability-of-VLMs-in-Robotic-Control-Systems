import json
import os
import sys
import time
from collections import defaultdict

import cv2


def frames_for(scenario):
    out = []
    d = "live_disruption_frames"
    for phase in ("baseline", "disrupt", "recover"):
        got = []
        for fn in sorted(os.listdir(d)):
            if fn.startswith(f"{scenario}_{phase}_") and fn.endswith(".jpg"):
                got.append(os.path.join(d, fn))
        out += got
    return out


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    scenario = "occlusion_full"
    paths = frames_for(scenario)
    if not paths:
        raise SystemExit("ABORT: no frames")

    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier
    from yolo_tracker import YoloTracker

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    verifier = make_fast_verifier()
    probe_v = getattr(verifier, "generate", None)

    times = defaultdict(float)
    counts = defaultdict(int)

    def timed(name, fn):
        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                times[name] += time.perf_counter() - t0
                counts[name] += 1
        return wrapper

    pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                            use_region_probe=True, probe_verifier=probe_v)
    pipe._detect = timed("detect", pipe._detect)
    pipe._verify = timed("verify", pipe._verify)
    pipe._diagnose = timed("diagnose", pipe._diagnose)
    pipe.probe_region = timed("probe", pipe.probe_region)

    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]
    print(f"{scenario}: {len(imgs)} frames, target={target!r}")

    pipe.step(imgs[0])
    times.clear(); counts.clear()

    t0 = time.perf_counter()
    for img in imgs[1:]:
        pipe.step(img)
    total = time.perf_counter() - t0
    n = len(imgs) - 1

    print(f"\n{'stage':<12}{'calls':>7}{'calls/frame':>13}"
          f"{'ms/call':>10}{'ms/frame':>11}{'% total':>9}")
    print("-" * 62)
    acc = 0.0
    for k in ("detect", "verify", "diagnose", "probe"):
        if not counts[k]:
            print(f"{k:<12}{0:>7}{0:>13.2f}{'-':>10}{0:>11.2f}{0:>9.1f}")
            continue
        ms_call = times[k] / counts[k] * 1e3
        ms_frame = times[k] / n * 1e3
        acc += times[k]
        print(f"{k:<12}{counts[k]:>7}{counts[k] / n:>13.2f}{ms_call:>10.1f}"
              f"{ms_frame:>11.2f}{times[k] / total * 100:>9.1f}")
    other = total - acc
    print(f"{'other':<12}{'':>7}{'':>13}{'':>10}{other / n * 1e3:>11.2f}"
          f"{other / total * 100:>9.1f}")
    print("-" * 62)
    print(f"{'TOTAL':<12}{'':>7}{'':>13}{'':>10}{total / n * 1e3:>11.2f}"
          f"{100.0:>9.1f}")
    print(f"\nthroughput {n / total:.2f} fps over {n} frames ({total:.1f}s)")

    out = {"scenario": scenario, "frames": n, "total_s": round(total, 3),
           "fps": round(n / total, 3),
           "stages": {k: {"calls": counts[k], "total_s": round(times[k], 4),
                          "ms_per_call": round(times[k] / counts[k] * 1e3, 2)
                          if counts[k] else None,
                          "ms_per_frame": round(times[k] / n * 1e3, 3)}
                      for k in ("detect", "verify", "diagnose", "probe")}}
    json.dump(out, open("profile_pipeline.json", "w"), indent=2)
    print("wrote profile_pipeline.json")


if __name__ == "__main__":
    main()
