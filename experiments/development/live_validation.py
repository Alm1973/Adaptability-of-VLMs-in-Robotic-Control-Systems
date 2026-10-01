import time

import cv2

from camera import Camera
from open_vocab_detect import (SCENE_DIFF_THRESHOLD, SceneGatedDetector,
                               scene_diff)

QUERIES = ["water bottle", "keyboard", "computer mouse", "laptop"]
SETTLE_READS = 15
ITERATIONS = 20
POLL_SLEEP = 0.5


def main():
    print("LIVE VALIDATION (camera only, no arm)")
    print(f"  queries: {QUERIES}")
    print(f"  scene-gate threshold: {SCENE_DIFF_THRESHOLD} "
          f"(static band measured on saved frames: max 7.27)\n")

    cam = Camera()
    print("settling auto-exposure...")
    prev = None
    for _ in range(SETTLE_READS):
        ok, f = cam.read()
        time.sleep(0.06)
    if not ok or f is None:
        print("CAMERA NOT DELIVERING FRAMES -- aborting")
        cam.release()
        return
    print(f"frame shape: {f.shape}\n")

    diffs = []
    base = f
    for _ in range(8):
        ok, cur = cam.read()
        if ok and cur is not None:
            diffs.append(scene_diff(base, cur))
        time.sleep(0.15)
    if diffs:
        print("real static-scene diffs (camera noise + AE drift only):")
        print(f"  n={len(diffs)} min={min(diffs):.2f} "
              f"mean={sum(diffs)/len(diffs):.2f} max={max(diffs):.2f}")
        verdict = ("OK - real noise sits inside the static band"
                   if max(diffs) < SCENE_DIFF_THRESHOLD else
                   "PROBLEM - real noise EXCEEDS the gate threshold; scene "
                   "gating will re-detect constantly and the latency win is "
                   "lost. Raise SCENE_DIFF_THRESHOLD (headroom: real changes "
                   "measured >= 42).")
        print(f"  -> {verdict}\n")

    det = SceneGatedDetector(QUERIES)
    lat_detect, degen = [], 0
    for i in range(1, ITERATIONS + 1):
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(POLL_SLEEP)
            continue
        t0 = time.time()
        result, fresh = det.detect(frame)
        dt = time.time() - t0
        if fresh:
            lat_detect.append(dt)
        present = [q for q in QUERIES if result.get(q)]
        if len(present) == len(QUERIES):
            degen += 1
        print(f"  {i:2d}/{ITERATIONS} {'DETECT' if fresh else 'cached'} "
              f"{dt:5.2f}s  visible: {present if present else '(none)'}")
        time.sleep(POLL_SLEEP)

    cam.release()
    s = det.stats
    total = s["detections"] + s["skips"]
    print("\n===== LIVE VALIDATION RESULT =====")
    print(f"detections run    : {s['detections']}")
    print(f"skipped (cached)  : {s['skips']} "
          f"({100*s['skips']/total:.0f}% of {total})")
    if lat_detect:
        print(f"detect latency    : mean {sum(lat_detect)/len(lat_detect):.2f}s "
              f"max {max(lat_detect):.2f}s")
    print(f"suspect-degenerate: {degen} (all-queries-true responses)")
    if diffs:
        print(f"real static diff  : max {max(diffs):.2f} vs threshold "
              f"{SCENE_DIFF_THRESHOLD}")
    print("\nIf skip rate is high and latency ~3s on detect, the shipped "
          "stack behaves live as it did on saved frames.")


if __name__ == "__main__":
    main()
