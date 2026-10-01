import json
import os
import statistics
import sys
import time

import cv2

N = 12
WARMUP = 3
FRAME = "live_disruption_frames/impostor_baseline_030.jpg"


def bench(tracker, verifier, img, target, n=N):
    det, vlm = [], []
    for _ in range(n):
        t0 = time.time()
        tracker.model.predict(img, conf=0.1, verbose=False)
        det.append(time.time() - t0)
        t0 = time.time()
        verifier(f"Is there a {target} in this image?", img)
        vlm.append(time.time() - t0)
    return det, vlm


def stat(v):
    return {"median": round(statistics.median(v), 4),
            "mean": round(statistics.mean(v), 4),
            "min": round(min(v), 4), "max": round(max(v), 4)}


def cpu_busy():
    try:
        import urllib.request
        q = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8188/queue", timeout=10).read())
        return len(q.get("queue_running", [])) > 0
    except Exception:
        return None


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    if not os.path.exists(FRAME):
        raise SystemExit(f"ABORT: no {FRAME}")
    img = cv2.imread(FRAME)

    from yolo_tracker import YoloTracker
    from run_study import make_fast_verifier
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    verifier = make_fast_verifier()

    bench(tracker, verifier, img, target, WARMUP)

    phase = sys.argv[-1] if sys.argv[-1] in ("quiet", "busy") else "quiet"
    busy = cpu_busy()
    print(f"phase={phase}  comfyui_generating={busy}")
    det, vlm = bench(tracker, verifier, img, target)
    out = {"phase": phase, "comfyui_generating": busy,
           "detector": stat(det), "vlm": stat(vlm), "n": N}
    print(f"  detector  median {out['detector']['median']:.4f}s  "
          f"mean {out['detector']['mean']:.4f}s")
    print(f"  vlm       median {out['vlm']['median']:.4f}s  "
          f"mean {out['vlm']['mean']:.4f}s")

    path = f"contention_{phase}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
