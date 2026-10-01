import sys
import time

import cv2

from arm import Arm
from camera import Camera

MOVE_DEG = 12
SETTLE_S = 1.6
HOME = (90, 90)

QUIET_MULT = 2.5


def watch(cam, seconds):
    out = []
    prev = None
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, f = cam.read()
        if not ok or f is None:
            continue
        small = cv2.cvtColor(cv2.resize(f, (160, 120)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            out.append((time.time() - t0,
                        float(cv2.absdiff(small, prev).mean())))
        prev = small
    return out


def profile(name, cam, arm, dx, dy, quiet):
    print(f"\n{name}: commanding X:{dx:+d} Y:{dy:+d}  "
          f"(quiet threshold {quiet:.2f})")
    arm.update(dx, dy)
    if not arm.connection_healthy:
        raise RuntimeError("arm link unhealthy")
    trace = watch(cam, SETTLE_S)

    moving = [(t, d) for t, d in trace if d > quiet]
    peak = max((d for _, d in trace), default=0.0)
    dur = (moving[-1][0] - moving[0][0]) if len(moving) > 1 else 0.0
    print(f"  {len(trace)} frames, peak diff {peak:.2f}, "
          f"{len(moving)} frames above quiet threshold, "
          f"motion spread over {dur:.2f}s")
    bar_max = max(peak, 1e-6)
    for t, d in trace[:26]:
        print(f"    t={t:4.2f}s  {d:6.2f}  " + "#" * int(38 * d / bar_max))
    return trace, moving, dur


def main():
    print("=" * 66)
    print(" SMOOTHNESS VERIFICATION -- THIS MOVES THE ARM (attended only)")
    print("=" * 66)

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never warmed up")
    print("opening arm on COM5 (resets the Uno -> centres at 90/90)...")
    arm = Arm()
    time.sleep(2.5)
    watch(cam, 0.8)

    baseline = watch(cam, 1.5)
    floor = (sum(d for _, d in baseline) / len(baseline)) if baseline else 1.0
    quiet = max(1.0, floor * QUIET_MULT)
    print(f"\nstatic noise floor: {floor:.2f} mean abs diff over "
          f"{len(baseline)} frames -> quiet threshold {quiet:.2f}")

    ok = True
    try:
        _, moving_big, dur_big = profile("LARGE STEP", cam, arm, MOVE_DEG, 0,
                                         quiet)
        time.sleep(0.6)
        _, moving_back, dur_back = profile("RETURN", cam, arm, -MOVE_DEG, 0,
                                           quiet)
        time.sleep(0.6)
        _, moving_small, dur_small = profile("SMALL CORRECTION", cam, arm, 2, 0,
                                             quiet)

        print("\n" + "=" * 66)
        print("VERDICT")
        print("=" * 66)
        if len(moving_big) >= 2:
            print(f"  large step: motion spans {len(moving_big)} frames over "
                  f"{dur_big:.2f}s -> EASED (a snap would be a single frame)")
        else:
            print(f"  !! large step occupied {len(moving_big)} frame(s) -- "
                  f"looks like a snap.\n     Easing may not be active; "
                  f"confirm the flash took.")
            ok = False

        if len(moving_small) >= 1:
            print(f"  small correction: {len(moving_small)} frames of motion "
                  f"over {dur_small:.2f}s -- a 2 deg nudge still moves")
        else:
            print("  !! a 2 deg correction produced NO visible motion -- the "
                  "easing may be\n     swallowing small commands, which would "
                  "break fine tracking.")
            ok = False

        print("\n" + ("PASS -- motion is eased and small corrections survive"
                      if ok else "FAIL -- check the firmware flash"))
    except Exception as e:
        print(f"\nABORT: {e}")
        ok = False
    finally:
        print("\nreturning to centre...")
        try:
            arm.update(arm.base - HOME[0], arm.tilt - HOME[1])
            time.sleep(1.2)
        except Exception as e:
            print(f"  (re-home failed: {e})")
        cam.release()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
