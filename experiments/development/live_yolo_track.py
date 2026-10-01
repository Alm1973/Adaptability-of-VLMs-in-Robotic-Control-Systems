import time

import cv2

from arm import Arm
from camera import Camera
from config import YOLO_TARGET
from controller import Controller
from yolo_tracker import YoloTracker

MAX_FRAMES = 150
MAX_SECONDS = 45.0
MAX_TOTAL_DEG = 120
WARMUP_TIMEOUT_S = 8.0
BLANK_MEAN_THRESHOLD = 1.0
RE_HOME = True


def main():
    print("=" * 72)
    print(f"BOUNDED LIVE YOLO TRACKING -- target={YOLO_TARGET!r}")
    print(f"limits: {MAX_FRAMES} frames / {MAX_SECONDS}s / {MAX_TOTAL_DEG} deg")
    print("=" * 72)

    cam = Camera()
    print("[CAM] opened; warming up...")
    frame = None
    warm_t0 = time.time()
    reads = 0
    while time.time() - warm_t0 < WARMUP_TIMEOUT_S:
        ok, f = cam.read()
        reads += 1
        if ok and f is not None and f.mean() >= BLANK_MEAN_THRESHOLD:
            frame = f
            break
        time.sleep(0.15)
    if frame is None:
        cam.release()
        raise SystemExit(
            f"ABORT: camera still blank after {reads} reads over "
            f"{WARMUP_TIMEOUT_S}s. Refusing to drive servos against no image.")
    print(f"[CAM] warm after {reads} reads / {time.time() - warm_t0:.1f}s")
    h, w = frame.shape[:2]
    print(f"[CAM] live {w}x{h}, mean={frame.mean():.1f}")

    tracker = YoloTracker(default_target=YOLO_TARGET)
    tracker.set_targets([YOLO_TARGET])
    controller = Controller()

    arm = Arm()
    start_base, start_tilt = arm.base, arm.tilt
    print(f"[ARM] connected; start position base={start_base} tilt={start_tilt}")

    cx0, cy0 = w // 2, h // 2
    frames = hits = commands = 0
    total_deg = 0
    t0 = time.time()
    stop_reason = "frame cap"

    try:
        while True:
            if frames >= MAX_FRAMES:
                stop_reason = "frame cap"
                break
            elapsed = time.time() - t0
            if elapsed >= MAX_SECONDS:
                stop_reason = "time cap"
                break
            if total_deg >= MAX_TOTAL_DEG:
                stop_reason = "movement budget"
                break

            ok, frame = cam.read_fresh()
            if not ok or frame is None:
                print("[CAM] read failed -> stopping")
                stop_reason = "camera read failure"
                break
            frames += 1

            r = tracker.find(frame)
            if r is None:
                controller.reset()
                continue
            hits += 1

            cx, cy = r["center"]
            errX, errY = cx - cx0, cy - cy0
            moveX, moveY = controller.compute(errX, errY)
            if moveX == 0 and moveY == 0:
                print(f"  f{frames:3d} {r['label']} conf={r['conf']:.2f} "
                      f"err=({errX:+5d},{errY:+5d}) -> in deadzone, no move")
                continue

            try:
                arm.update(moveX, moveY)
            except Exception as e:
                print(f"[ARM] write FAILED: {e}")
                stop_reason = "arm write failure"
                break

            commands += 1
            total_deg += abs(moveX) + abs(moveY)
            print(f"  f{frames:3d} {r['label']} conf={r['conf']:.2f} "
                  f"err=({errX:+5d},{errY:+5d}) -> move=({moveX:+d},{moveY:+d}) "
                  f"pos=({arm.base},{arm.tilt}) budget={total_deg}/{MAX_TOTAL_DEG}")
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
    finally:
        elapsed = time.time() - t0
        if RE_HOME and arm.connection_healthy:
            db, dt = arm.base - start_base, arm.tilt - start_tilt
            if db or dt:
                print(f"[ARM] re-homing by ({db:+d},{dt:+d}) -> "
                      f"({start_base},{start_tilt})")
                try:
                    arm.update(db, dt)
                except Exception as e:
                    print(f"[ARM] re-home failed: {e}")
        cam.release()
        print("\n" + "=" * 72)
        print(f"stopped: {stop_reason}")
        print(f"  frames      {frames}")
        print(f"  detections  {hits}  ({100.0 * hits / frames:.0f}%)"
              if frames else "  detections  0")
        print(f"  commands    {commands}")
        print(f"  degrees     {total_deg} / {MAX_TOTAL_DEG}")
        print(f"  elapsed     {elapsed:.1f}s  ({frames / elapsed:.1f} fps)"
              if elapsed > 0 else "")
        print(f"  arm final   base={arm.base} tilt={arm.tilt} "
              f"(started {start_base},{start_tilt})")
        print("=" * 72)


if __name__ == "__main__":
    main()
