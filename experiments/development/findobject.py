import sys
import time

import cv2

from camera import Camera
from open_vocab_detect import VOTE_SAMPLES, detect

SETTLE_READS = 12
QUIT_WORDS = {"q", "quit", "exit", ""}


def grab(cam):
    frame = None
    for _ in range(SETTLE_READS):
        ok, f = cam.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.05)
    return frame


def look_for(cam, queries):
    frame = grab(cam)
    if frame is None:
        print("  !! camera returned no frame")
        return
    t0 = time.time()
    results, lists = detect(frame, queries, samples=VOTE_SAMPLES)
    dt = time.time() - t0

    print()
    for q in queries:
        if results.get(q):
            print(f"  ✅ FOUND: {q}")
        else:
            print(f"  ❌ not found: {q}")
    seen = []
    for l in lists:
        for it in l:
            if it.lower() not in [s.lower() for s in seen]:
                seen.append(it)
    print(f"\n  (it saw: {', '.join(seen) if seen else 'nothing'})")
    print(f"  ({dt:.1f}s, {len(lists)} looks)\n")


def main():
    args = [a for a in sys.argv[1:] if a.strip()]
    print("=" * 58)
    print(" FIND OBJECT — type any object name, the robot looks for it")
    print("=" * 58)
    print(" Type several separated by commas.  'q' or Enter to quit.\n")

    print("opening camera...")
    try:
        cam = Camera()
    except Exception as e:
        print(f"\nCamera failed to open: {e}")
        print("Is the C270 plugged in? Run:  python camera_probe.py")
        return
    grab(cam)
    print("ready.\n")

    try:
        if args:
            look_for(cam, args)
            return
        while True:
            try:
                raw = input("what should I look for? > ").strip()
            except EOFError:
                break
            if raw.lower() in QUIT_WORDS:
                break
            queries = [q.strip() for q in raw.split(",") if q.strip()]
            if not queries:
                continue
            print(f"looking for {queries} ...")
            look_for(cam, queries)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        print("camera released. bye.")


if __name__ == "__main__":
    main()
