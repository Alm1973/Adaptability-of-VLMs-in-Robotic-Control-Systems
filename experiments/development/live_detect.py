import sys
import time

import cv2

from camera import Camera
from open_vocab_detect import SceneGatedDetector

DEFAULT_QUERIES = ["water bottle", "keyboard", "computer mouse", "laptop"]
SETTLE_READS = 8
POLL_SLEEP = 0.4


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES
    print("live open-vocabulary detection")
    print(f"  watching for: {queries}")
    print("  (scene-gated: inference only runs when the view changes)")
    print("  Ctrl+C to stop\n")

    cam = Camera()
    for _ in range(SETTLE_READS):
        cam.read()
        time.sleep(0.05)

    det = SceneGatedDetector(queries)
    last_line = None
    t_start = time.time()
    frames = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(POLL_SLEEP)
                continue
            frames += 1
            t0 = time.time()
            result, fresh = det.detect(frame)
            dt = time.time() - t0

            present = [q for q in queries if result.get(q)]
            line = ", ".join(present) if present else "(nothing from the list)"
            if fresh or line != last_line:
                stamp = time.strftime("%H:%M:%S")
                tag = f"DETECT {dt:4.1f}s" if fresh else "cached      "
                print(f"[{stamp}] {tag}  visible: {line}")
                last_line = line
            time.sleep(POLL_SLEEP)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        cam.release()
        elapsed = time.time() - t_start
        s = det.stats
        total = s["detections"] + s["skips"]
        if total:
            print(f"\nframes processed : {frames}")
            print(f"detections run   : {s['detections']}")
            print(f"skipped (cached) : {s['skips']} "
                  f"({100*s['skips']/total:.0f}%)")
            print(f"elapsed          : {elapsed:.0f}s")
            print(f"inference avoided by scene gating: "
                  f"~{s['skips']*3.2:.0f}s")


if __name__ == "__main__":
    main()
