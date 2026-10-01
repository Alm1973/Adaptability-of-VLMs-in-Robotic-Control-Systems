import cv2
import threading
import time
from datetime import datetime
from camera import Camera
from tracker import Tracker
from controller import Controller
from arm import Arm
from tile_map import TileMap
from vlm_recovery import get_recovery_action, warm_up, force_reload_model, get_verification_hybrid
from config import TILE_CAPTURE_INTERVAL_SEC, USE_YOLO_TRACKER, YOLO_TARGET

TRACKING = "TRACKING"
REACQUIRE = "REACQUIRE"

MISSES_BEFORE_REACQUIRE = 10
RECOVERY_STEP = 15
MAX_SAME_ACTION_REPEATS = 3
BOUND_MARGIN = 5
RELOAD_COOLDOWN_SECONDS = 60

ALL_DIRECTIONS = ["scan_left", "scan_right", "tilt_up", "tilt_down"]

cam = Camera()
if USE_YOLO_TRACKER:
    from yolo_tracker import YoloTracker
    tracker = YoloTracker(default_target=YOLO_TARGET)
else:
    tracker = Tracker()
controller = Controller()
arm = Arm()
tile_map = TileMap()

warm_up()

state = TRACKING
miss_count = 0

last_raw_action = None
raw_repeat_count = 0

scan_counts = {d: 0 for d in ALL_DIRECTIONS}

last_known_direction = None

currently_tracking = False

vlm_thread = None
vlm_result = {"action": None, "ready": False}
reacquire_generation = 0

reload_thread = None
last_reload_time = 0.0
clean_calls_since_reload = 0

tile_thread = None
last_tile_capture_time = 0.0

verification_thread = None
verification_generation = 0
verification_result = {"outcome": None, "ready": False, "center": None}


def safe_arm_update(moveX, moveY):
    try:
        arm.update(moveX, moveY)
    except Exception as e:
        print(f"[ARM] Write failed, skipping this command: {e}")


def build_scan_history_text():
    tried = [f"{d} ({c}x)" for d, c in scan_counts.items() if c > 0]
    untried = [d for d, c in scan_counts.items() if c == 0]
    parts = []
    if tried:
        parts.append("Already tried: " + ", ".join(tried) + ".")
    if untried:
        parts.append("NOT yet tried: " + ", ".join(untried) + ".")
    return " ".join(parts) if parts else None


def least_explored_direction():
    return min(ALL_DIRECTIONS, key=lambda d: scan_counts[d])


def vlm_worker(scan_history, last_known_direction, generation):
    mosaic = tile_map.build_mosaic()
    action = get_recovery_action(
        mosaic, scan_history=scan_history, last_known_direction=last_known_direction
    )
    if generation != reacquire_generation:
        print(f"[VLM] Discarding stale result from generation {generation} "
              f"(current is {reacquire_generation})")
        return
    vlm_result["action"] = action
    vlm_result["ready"] = True


def reload_worker():
    global clean_calls_since_reload
    force_reload_model()
    clean_calls_since_reload = 0


def maybe_force_reload():
    global reload_thread, last_reload_time

    if reload_thread is not None and reload_thread.is_alive():
        print("[RELOAD] Reload already in progress, skipping")
        return

    now = time.time()
    if now - last_reload_time < RELOAD_COOLDOWN_SECONDS:
        remaining = RELOAD_COOLDOWN_SECONDS - (now - last_reload_time)
        print(f"[RELOAD] Skipping reload, cooldown active ({remaining:.0f}s remaining)")
        return

    last_reload_time = now
    reload_thread = threading.Thread(target=reload_worker, daemon=True)
    reload_thread.start()


def tile_capture_worker(frame, base_angle, tilt_angle):
    bucket = tile_map.update(frame, base_angle, tilt_angle)
    coverage_pct = tile_map.coverage() * 100
    print(f"[TILEMAP] Captured tile at bucket {bucket}, coverage now {coverage_pct:.1f}%")


def maybe_capture_tile(frame, base_angle, tilt_angle):
    global tile_thread, last_tile_capture_time

    if tile_thread is not None and tile_thread.is_alive():
        return

    now = time.time()
    if now - last_tile_capture_time < TILE_CAPTURE_INTERVAL_SEC:
        return

    last_tile_capture_time = now
    tile_thread = threading.Thread(
        target=tile_capture_worker,
        args=(frame, base_angle, tilt_angle),
        daemon=True,
    )
    tile_thread.start()


def verification_worker(frame, box, center, contour, generation):
    outcome = get_verification_hybrid(frame, box, contour)
    if generation != verification_generation:
        print(f"[VERIFY] Discarding stale result from generation {generation} "
              f"(current is {verification_generation})")
        return
    verification_result["outcome"] = outcome
    verification_result["center"] = center
    verification_result["ready"] = True


def maybe_trigger_verification(frame, box, center, contour):
    global verification_thread, verification_generation

    verification_generation += 1
    verification_thread = threading.Thread(
        target=verification_worker,
        args=(frame, box, center, contour, verification_generation),
        daemon=True,
    )
    verification_thread.start()


def save_mosaic_snapshot():
    mosaic = tile_map.build_mosaic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mosaic_{timestamp}.jpg"
    cv2.imwrite(filename, mosaic)
    coverage_pct = tile_map.coverage() * 100
    print(f"[TILEMAP] Saved mosaic snapshot to {filename} (coverage {coverage_pct:.1f}%)")


try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera error")
            break

        clean_frame = frame.copy()

        h, w = frame.shape[:2]
        centerX = w // 2
        centerY = h // 2

        cv2.drawMarker(frame, (centerX, centerY), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

        result = tracker.find_red(frame)

        if verification_result["ready"]:
            outcome = verification_result["outcome"]
            rejected_center = verification_result["center"]
            print(f"[VERIFY] Result: {outcome}")
            verification_result = {"outcome": None, "ready": False, "center": None}

            if outcome == "other":
                print(f"[VERIFY] Rejected detection at {rejected_center} -- "
                      f"forcing REACQUIRE")
                tracker.add_suppression_zone(rejected_center)
                state = REACQUIRE
                miss_count = 0
                last_raw_action = None
                raw_repeat_count = 0
                scan_counts = {d: 0 for d in ALL_DIRECTIONS}
                last_known_direction = None
                currently_tracking = False
                tracker.reset_history()
                vlm_thread = None
                vlm_result = {"action": None, "ready": False}
                cv2.imshow("AVI Vision", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('m'):
                    save_mosaic_snapshot()
                continue

        if state == TRACKING:
            maybe_capture_tile(clean_frame, arm.base, arm.tilt)

            if result:
                if not currently_tracking:
                    maybe_trigger_verification(clean_frame, result["box"],
                                               result["center"], result["contour"])
                currently_tracking = True

                miss_count = 0
                last_raw_action = None
                raw_repeat_count = 0
                scan_counts = {d: 0 for d in ALL_DIRECTIONS}

                x, y, bw, bh = result["box"]
                cx, cy = result["center"]

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                errorX = cx - centerX
                errorY = cy - centerY

                moveX, moveY = controller.compute(errorX, errorY)
                safe_arm_update(moveX, moveY)

                cv2.putText(frame, f"ErrX:{errorX} ErrY:{errorY}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MoveX:{moveX} MoveY:{moveY}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "STATE: TRACKING", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                print(f"ERROR: {errorX}, {errorY} | MOVE: {moveX}, {moveY}")
            else:
                currently_tracking = False
                miss_count += 1
                cv2.putText(frame, f"STATE: TRACKING (miss {miss_count}/{MISSES_BEFORE_REACQUIRE})",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if miss_count >= MISSES_BEFORE_REACQUIRE:
                    last_known_direction = tracker.get_last_known_direction()
                    print(f"[STATE] Lost tracking -> entering REACQUIRE "
                          f"(last_known_direction={last_known_direction})")
                    state = REACQUIRE
                    vlm_thread = None
                    vlm_result = {"action": None, "ready": False}

        elif state == REACQUIRE:
            if result:
                print("[STATE] Object reacquired (OpenCV) -> back to TRACKING")
                state = TRACKING
                miss_count = 0
                last_raw_action = None
                raw_repeat_count = 0
                scan_counts = {d: 0 for d in ALL_DIRECTIONS}
                last_known_direction = None
                currently_tracking = False
                tracker.reset_history()
                vlm_thread = None
                vlm_result = {"action": None, "ready": False}
                cv2.imshow("AVI Vision", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('m'):
                    save_mosaic_snapshot()
                continue

            cv2.putText(frame, "STATE: REACQUIRE (asking VLM...)", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if vlm_thread is None:
                reacquire_generation += 1
                scan_history = build_scan_history_text()
                vlm_thread = threading.Thread(
                    target=vlm_worker,
                    args=(scan_history, last_known_direction, reacquire_generation),
                    daemon=True,
                )
                vlm_thread.start()

            if vlm_result["ready"]:
                raw_action = vlm_result["action"]
                print(f"[VLM] Raw action: {raw_action}")

                if raw_action is None:
                    print(f"[VLM] Degenerate after {clean_calls_since_reload} "
                          f"clean calls since last reload")
                    clean_calls_since_reload = 0
                else:
                    clean_calls_since_reload += 1
                    print(f"[VLM] Clean calls since last reload: {clean_calls_since_reload}")

                vlm_thread = None
                vlm_result = {"action": None, "ready": False}

                if raw_action == "object_found":
                    print("[STATE] Object reacquired (VLM) -> back to TRACKING")
                    state = TRACKING
                    miss_count = 0
                    last_raw_action = None
                    raw_repeat_count = 0
                    scan_counts = {d: 0 for d in ALL_DIRECTIONS}
                    last_known_direction = None
                    currently_tracking = False
                    tracker.reset_history()
                else:
                    if raw_action == last_raw_action:
                        raw_repeat_count += 1
                    else:
                        raw_repeat_count = 1
                    last_raw_action = raw_action

                    action = raw_action

                    if raw_repeat_count > MAX_SAME_ACTION_REPEATS:
                        if raw_action is None:
                            maybe_force_reload()
                        fallback = least_explored_direction()
                        print(f"[REPEAT] VLM stuck on '{raw_action}' x{raw_repeat_count}, "
                              f"forcing least-explored direction: {fallback}")
                        action = fallback
                        last_raw_action = action
                        raw_repeat_count = 1

                    at_left_limit = arm.base <= BOUND_MARGIN
                    at_right_limit = arm.base >= (180 - BOUND_MARGIN)
                    at_top_limit = arm.tilt <= BOUND_MARGIN
                    at_bottom_limit = arm.tilt >= (180 - BOUND_MARGIN)

                    if action == "scan_left" and at_left_limit:
                        print(f"[BOUND] At left limit, overriding to scan_right")
                        action = "scan_right"
                    elif action == "scan_right" and at_right_limit:
                        print(f"[BOUND] At right limit, overriding to scan_left")
                        action = "scan_left"
                    elif action == "tilt_up" and at_top_limit:
                        print(f"[BOUND] At top limit, overriding to tilt_down")
                        action = "tilt_down"
                    elif action == "tilt_down" and at_bottom_limit:
                        print(f"[BOUND] At bottom limit, overriding to tilt_up")
                        action = "tilt_up"

                    if action in scan_counts:
                        scan_counts[action] += 1

                    if action == "scan_left":
                        safe_arm_update(-RECOVERY_STEP, 0)
                    elif action == "scan_right":
                        safe_arm_update(RECOVERY_STEP, 0)
                    elif action == "tilt_up":
                        safe_arm_update(0, -RECOVERY_STEP)
                    elif action == "tilt_down":
                        safe_arm_update(0, RECOVERY_STEP)

        cv2.imshow("AVI Vision", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            save_mosaic_snapshot()
finally:
    cam.release()
    cv2.destroyAllWindows()
    save_mosaic_snapshot()