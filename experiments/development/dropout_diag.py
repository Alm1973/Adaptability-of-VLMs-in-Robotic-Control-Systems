import time
from collections import Counter

from camera import Camera
from config import YOLO_TARGET
from yolo_tracker import YoloTracker, MIN_CONFIDENCE, MIN_CONSECUTIVE_HITS

FRAMES = 150


def main():
    print("=" * 72)
    print(f"DROPOUT DIAGNOSTIC -- target={YOLO_TARGET!r}, {FRAMES} frames")
    print(f"threshold={MIN_CONFIDENCE}  streak={MIN_CONSECUTIVE_HITS}")
    print("NO ARM -- camera only")
    print("=" * 72)

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")

    t = YoloTracker(default_target=YOLO_TARGET, debug=True)
    t.set_targets([YOLO_TARGET])

    reported = 0
    t0 = time.time()
    for i in range(FRAMES):
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            print("camera read failed"); break
        if t.find(frame) is not None:
            reported += 1
    cam.release()
    elapsed = time.time() - t0

    log = t.debug_log
    n = len(log)
    if not n:
        raise SystemExit("no frames logged")

    seen_any = sum(1 for r in log if r["best_raw"] > 0)
    over = sum(1 for r in log if r["over_threshold"])
    confs = [r["best_raw"] for r in log if r["best_raw"] > 0]
    confs.sort()

    print(f"\nframes                       {n}  ({n / elapsed:.1f} fps)")
    print(f"target seen at ALL (>={t.DEBUG_FLOOR})  {seen_any}")
    print(f"target OVER threshold {MIN_CONFIDENCE}    {over}")
    print(f"actually REPORTED by find()  {reported}")
    if confs:
        print(f"\nraw confidence when seen: min={confs[0]:.3f} "
              f"p50={confs[len(confs) // 2]:.3f} max={confs[-1]:.3f}")
        buckets = Counter()
        for c in confs:
            buckets[f"{int(c * 10) / 10:.1f}"] += 1
        for b in sorted(buckets):
            print(f"    {b}-{float(b) + 0.1:.1f}: {'#' * buckets[b]} "
                  f"({buckets[b]})")

    delta = over - reported
    if t.hold_confidence < MIN_CONFIDENCE:
        print(f"\nhysteresis ACTIVE (acquire {MIN_CONFIDENCE} / hold "
              f"{t.hold_confidence}); reported-minus-over = {-delta:+d} "
              f"frames held on the loose bar")
    else:
        print(f"\nover threshold but NOT reported: {delta}"
              f"   <- lost to the consecutive-hit gate")

    print("\n" + "-" * 72)
    if over == 0 and seen_any > 0:
        print("VERDICT (A): CONFIDENCE. The target is visible but never clears "
              f"{MIN_CONFIDENCE}. Threshold/pose problem, not the streak gate.")
    elif lost_to_streak > over * 0.3:
        print("VERDICT (B): STREAK GATE. Raw confidence clears threshold often "
              "but FLICKERS, so the consecutive-hit requirement keeps "
              "resetting. Needs hysteresis, not a lower threshold.")
    elif seen_any == 0:
        print("VERDICT (C): NOT IN FRAME. The target is not visible at all -- "
              "not a tuning problem, the camera is pointed elsewhere.")
    else:
        print("VERDICT: mostly clean -- detection is working at this pose.")
    print("-" * 72)


if __name__ == "__main__":
    main()
