
import time
import cv2

from tracker import Tracker
from vlm_recovery import get_verification_hybrid, warm_up
import llm_backend

CASES = [
    ("obj2_06_far_left.jpg", "other", "canyon wallpaper on a monitor"),
    ("obj2_07_left_low.jpg", "cup", "stacked Solo cups, edge/occluded"),
    ("obj2_08_left_lower.jpg", "cup", "red cup, top of frame"),
    ("obj2_09_center_low.jpg", "cup", "cup edge sliver 81x66 (marginal)"),
    ("obj2_11_far_right.jpg", "other", "tinsel/streamer blob"),
    ("obj2_12_right_high.jpg", "other", "wall streamer"),
    ("obj2_13_right_higher.jpg", "other", "streamer knot"),
    ("index1_test2.jpg", "cup", "clean cup (reference)"),
    ("hand_test.jpg", "cup", "held cup stack (reference)"),
]

BASELINE_VLM_ONLY = "6/9 (session 5, CPU placement)"
BASELINE_CLASSICAL = "8/9 (session 5, rule applied on paper)"


def main():
    warm_up()

    results = []
    for path, truth, note in CASES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: FAILED TO LOAD")
            continue
        result = Tracker().find_red(frame)
        if result is None:
            print(f"{path}: no red detection (unexpected -- was detected in session 5)")
            results.append((path, truth, None, "NO_DETECTION", 0.0))
            continue

        t0 = time.time()
        verdict = get_verification_hybrid(frame, result["box"], result["contour"])
        elapsed = time.time() - t0
        correct = (verdict == truth)
        results.append((path, truth, verdict, correct, elapsed))
        print(f"{path}: truth={truth} verdict={verdict} "
              f"{'OK' if correct else 'WRONG'} ({elapsed:.1f}s) -- {note}")

    print("\n=== INTEGRATION RESULTS ===")
    n = sum(1 for r in results if r[3] in (True, False))
    n_ok = sum(1 for r in results if r[3] is True)
    print(f"{'image':<28} {'truth':<7} {'verdict':<9} {'correct':<8} {'latency'}")
    for path, truth, verdict, correct, elapsed in results:
        print(f"{path:<28} {truth:<7} {str(verdict):<9} {str(correct):<8} {elapsed:.1f}s")
    print(f"\nHybrid pipeline: {n_ok}/{n} correct")
    print(f"Baseline VLM-only:   {BASELINE_VLM_ONLY}")
    print(f"Baseline classical:  {BASELINE_CLASSICAL}")
    print(f"PASS criterion (>= 8/9): {'PASS' if n_ok >= 8 else 'FAIL'}")

    llm_backend.shutdown()


if __name__ == "__main__":
    main()
