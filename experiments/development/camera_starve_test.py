import sys
import time

import cv2


def sample(cam, seconds, label, work=None):
    ok_n = fail_n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cam.read()
        if ok and frame is not None:
            ok_n += 1
        else:
            fail_n += 1
        if work is not None:
            work(frame)
    el = time.time() - t0
    rate = ok_n / el
    print(f"  {label:<34} ok={ok_n:<5} fail={fail_n:<5} {rate:5.1f} fps")
    return {"ok": ok_n, "fail": fail_n, "fps": round(rate, 2)}


def main():
    from camera import Camera
    use_arm = "--arm" in sys.argv

    cam = Camera()
    if not cam.warm:
        raise SystemExit("ABORT: camera never warmed")
    print("camera warm\n")

    print("A. baseline, nothing else running")
    sample(cam, 6, "fresh camera")

    print("\nB. after sitting open")
    time.sleep(5)
    sample(cam, 6, "after 5s idle")

    print("\nC. with the GPU stack loaded and running")
    from yolo_tracker import YoloTracker
    from run_study import make_fast_verifier
    t = YoloTracker(default_target="red cup")
    t.set_targets(["red cup"], allow_unreliable=True, quiet=True)
    v = make_fast_verifier()

    def gpu_work(frame):
        if frame is not None:
            t.model.predict(frame, conf=0.1, verbose=False)

    sample(cam, 8, "reads + detector every frame", gpu_work)

    def gpu_heavy(frame):
        if frame is not None:
            t.model.predict(frame, conf=0.1, verbose=False)
            v("Is there a red cup in this image?", frame)

    sample(cam, 8, "reads + detector + VLM every frame", gpu_heavy)

    if use_arm:
        print("\nD. across arm motion  ⚠️ THE ARM WILL MOVE")
        from arm import Arm
        arm = Arm()
        time.sleep(2.5)
        home = (arm.base, arm.tilt)
        try:
            sample(cam, 5, "arm idle")
            arm.update(15, 0)
            sample(cam, 5, "immediately after a 15deg move")
            arm.update(-15, 0)
            sample(cam, 5, "after returning")
        finally:
            db, dt = home[0] - arm.base, home[1] - arm.tilt
            if db or dt:
                arm.update(db, dt)
                time.sleep(1.2)
            print("  arm re-homed")

    cam.release()
    cv2.destroyAllWindows()
    print("\nIf A is healthy (~15-30 fps) and C collapses, the cause is GPU")
    print("contention starving the capture thread. If B collapses, the camera")
    print("times out when not read continuously. If only D collapses, it is")
    print("the arm -- USB power or bus contention.")


if __name__ == "__main__":
    main()
