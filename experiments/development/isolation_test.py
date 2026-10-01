
import os
import cv2

from vlm_recovery import get_recovery_action, warm_up

TEST_IMAGE = "index1_test2.jpg"
NUM_CALLS = 20


def load_test_image():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_IMAGE)
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Failed to load test image: {path}")
    print(f"[ISOLATION] Using static test image: {TEST_IMAGE} "
          f"({frame.shape[1]}x{frame.shape[0]})")
    return frame


def main():
    frame = load_test_image()

    warm_up()

    degenerate_count = 0
    valid_actions = []

    for i in range(1, NUM_CALLS + 1):
        print(f"[ISOLATION] --- call {i}/{NUM_CALLS} ---")
        action = get_recovery_action(frame, scan_history=None, last_known_direction=None)

        if action is None:
            degenerate_count += 1
            print(f"[ISOLATION] call {i}/{NUM_CALLS}: DEGENERATE "
                  f"(raw text printed above by [VLM][DEGENERATE])")
        else:
            valid_actions.append(action)
            print(f"[ISOLATION] call {i}/{NUM_CALLS}: VALID -> {action}")

    valid_count = NUM_CALLS - degenerate_count

    print("\n=== SUMMARY ===")
    print(f"Degenerate: {degenerate_count}/{NUM_CALLS}")
    print(f"Valid:      {valid_count}/{NUM_CALLS}")
    print(f"Valid actions returned: {valid_actions}")


if __name__ == "__main__":
    main()
