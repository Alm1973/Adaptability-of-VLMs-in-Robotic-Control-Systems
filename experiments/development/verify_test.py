
import cv2

from tracker import Tracker
from vlm_recovery import get_verification, warm_up

TEST_CASES = [
    ("index1_test2.jpg", "cup", "clean cup + keyboard shot"),
    ("hand_test.jpg", "cup", "two stacked cups held in hand -- find_red() selects the cups, not the hand"),
]


def main():
    warm_up()

    results = []

    for path, expected, note in TEST_CASES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"SKIP {path}: failed to load")
            continue

        tracker = Tracker()
        result = tracker.find_red(frame)

        if result is None:
            print(f"SKIP {path}: find_red() found no contour (nothing to verify)")
            continue

        box = result["box"]
        print(f"{path}: find_red() selected box={box} area={result['area']:.0f}")

        outcome = get_verification(frame, box)
        correct = (outcome == expected)
        results.append((path, expected, outcome, correct, note))
        print(f"  -> expected={expected} actual={outcome} correct={correct}")

    print("\n=== RESULTS TABLE ===")
    print(f"{'image':<20} {'expected':<10} {'actual':<10} {'correct'}")
    for path, expected, outcome, correct, note in results:
        print(f"{path:<20} {expected:<10} {str(outcome):<10} {correct}")

    n = len(results)
    n_correct = sum(1 for r in results if r[3])
    pct = (n_correct / n * 100) if n else 0.0
    print(f"\n{n_correct}/{n} correct ({pct:.1f}%)")


if __name__ == "__main__":
    main()
