
import time
import cv2

from camera import Camera
from arm import Arm

REQUIRES_HUMAN_APPROVAL = True
HUMAN_APPROVAL_GRANTED = True

BOUND_MARGIN = 5

PHASE_A_WAYPOINTS = [
    (90, 90, "home"),
    (102, 90, "verify_base_plus"),
    (102, 102, "verify_tilt_plus"),
    (90, 90, "verify_return_home"),
]

PHASE_B_WAYPOINTS = [
    (65, 90, "left_mid"),
    (40, 90, "far_left"),
    (40, 115, "left_low"),
    (65, 130, "left_lower"),
    (90, 130, "center_low"),
    (115, 115, "right_low"),
    (140, 90, "far_right"),
    (140, 65, "right_high"),
    (115, 55, "right_higher"),
    (90, 65, "center_high"),
]

SETTLE_SECONDS = 1.2
WARMUP_READS = 3

FRAME_DIFF_STUCK_THRESHOLD = 6.0


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clamp(angle):
    return max(BOUND_MARGIN, min(180 - BOUND_MARGIN, angle))


def frame_diff(frame_a, frame_b):
    small_a = cv2.cvtColor(cv2.resize(frame_a, (160, 120)), cv2.COLOR_BGR2GRAY)
    small_b = cv2.cvtColor(cv2.resize(frame_b, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(small_a, small_b).mean())


def capture_frame(cam):
    frame = None
    for _ in range(WARMUP_READS):
        ret, frame = cam.read()
    return frame


def move_and_capture(cam, arm, target_base, target_tilt, label, index):
    target_base = clamp(target_base)
    target_tilt = clamp(target_tilt)
    delta_x = target_base - arm.base
    delta_y = target_tilt - arm.tilt

    print(f"[{ts()}] #{index} '{label}': target=({target_base},{target_tilt}) "
          f"delta=({delta_x},{delta_y}) tracked_before=({arm.base},{arm.tilt})")

    if delta_x != 0 or delta_y != 0:
        try:
            arm.update(delta_x, delta_y)
        except Exception as e:
            print(f"  Arm move failed: {e} -- one retry")
            time.sleep(1)
            try:
                arm.update(target_base - arm.base, target_tilt - arm.tilt)
            except Exception as e2:
                print(f"  Retry failed: {e2} -- ABORTING SEQUENCE")
                return None, None, f"ARM_ERROR: {e2}"

    if not arm.connection_healthy:
        return None, None, "ARM_CONNECTION_UNHEALTHY"

    time.sleep(SETTLE_SECONDS)

    try:
        frame = capture_frame(cam)
    except Exception as e:
        return None, None, f"CAMERA_ERROR: {e}"
    if frame is None:
        return None, None, "NO_FRAME"

    filename = f"obj2_{index:02d}_{label}.jpg"
    cv2.imwrite(filename, frame)
    print(f"  [{ts()}] saved {filename} (brightness={frame.mean():.1f}) "
          f"tracked_after=({arm.base},{arm.tilt})")
    return frame, filename, None


def main():
    if REQUIRES_HUMAN_APPROVAL and not HUMAN_APPROVAL_GRANTED:
        print("REFUSING TO RUN: human approval gate not granted. See file "
              "header and PROGRESS.md post-mortem.")
        return

    print(f"[{ts()}] Opening camera...")
    cam = Camera()
    time.sleep(1.0)
    for _ in range(WARMUP_READS):
        cam.read()

    print(f"[{ts()}] Opening arm connection...")
    arm = Arm()

    captured = []
    prev_frame = None
    low_diff_streak = 0
    aborted = None

    def run_waypoint(base, tilt, label, index, phase):
        nonlocal prev_frame, low_diff_streak, aborted
        moved = (clamp(base) != arm.base) or (clamp(tilt) != arm.tilt)
        frame, filename, error = move_and_capture(cam, arm, base, tilt, label, index)
        if error:
            aborted = error
            return False

        if prev_frame is not None and moved:
            diff = frame_diff(prev_frame, frame)
            print(f"  frame diff vs previous capture: {diff:.2f} "
                  f"(stuck threshold {FRAME_DIFF_STUCK_THRESHOLD})")
            if diff < FRAME_DIFF_STUCK_THRESHOLD:
                if phase == "A":
                    aborted = (f"POSITION_SANITY_FAIL: commanded move "
                               f"'{label}' produced near-identical frame "
                               f"(diff={diff:.2f}) during verification "
                               f"phase -- stuck-arm signature, stopping "
                               f"before the sweep")
                    return False
                low_diff_streak += 1
                if low_diff_streak >= 2:
                    aborted = (f"POSITION_SANITY_FAIL: two consecutive "
                               f"commanded moves produced near-identical "
                               f"frames (latest '{label}', diff="
                               f"{diff:.2f}) -- stuck-arm signature")
                    return False
                print("  WARNING: low frame diff after a commanded move -- "
                      "one more consecutive occurrence stops the sweep")
            else:
                low_diff_streak = 0

        prev_frame = frame
        captured.append((filename, label))
        return True

    index = 0
    print(f"\n[{ts()}] === PHASE A: small verification moves ===")
    for base, tilt, label in PHASE_A_WAYPOINTS:
        index += 1
        if not run_waypoint(base, tilt, label, index, "A"):
            break

    if aborted is None:
        print(f"\n[{ts()}] Phase A clean: {index} captures, position "
              f"tracking consistent with observed frame changes. "
              f"Proceeding to sweep.")
        print(f"\n[{ts()}] === PHASE B: reduced-envelope sweep ===")
        for base, tilt, label in PHASE_B_WAYPOINTS:
            index += 1
            if not run_waypoint(base, tilt, label, index, "B"):
                break

    if aborted is None and arm.connection_healthy:
        print(f"\n[{ts()}] Returning arm to home (90, 90)...")
        try:
            arm.update(90 - arm.base, 90 - arm.tilt)
        except Exception as e:
            print(f"  Return-home failed (not retried): {e}")
    elif aborted is not None:
        print(f"\n[{ts()}] NOT returning home -- sequence aborted "
              f"({aborted}). Arm's true position must be checked by eye.")

    cam.release()

    print(f"\n[{ts()}] === CAPTURE SUMMARY ===")
    for filename, label in captured:
        print(f"  {filename}  ({label})")
    if aborted:
        print(f"ABORTED: {aborted}")
    print(f"{len(captured)} captures, aborted={aborted is not None}")


if __name__ == "__main__":
    main()
