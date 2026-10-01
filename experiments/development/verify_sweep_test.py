
import cv2
import ollama

from tracker import Tracker
from vlm_recovery import (
    MODEL_NAME, OPTIONS, VERIFICATION_PROMPT, _crop_with_padding,
    extract_verification,
)

CPU_OPTIONS = dict(OPTIONS)
CPU_OPTIONS["num_gpu"] = 0

IMAGES = [
    "obj2_06_far_left.jpg",
    "obj2_07_left_low.jpg",
    "obj2_08_left_lower.jpg",
    "obj2_09_center_low.jpg",
    "obj2_11_far_right.jpg",
    "obj2_12_right_high.jpg",
    "obj2_13_right_higher.jpg",
    "index1_test2.jpg",
    "hand_test.jpg",
]


def main():
    for path in IMAGES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: FAILED TO LOAD")
            continue
        result = Tracker().find_red(frame)
        if result is None:
            print(f"{path}: no red detection (skipped)")
            continue

        crop = _crop_with_padding(frame, result["box"])
        small = cv2.resize(crop, (480, 360))
        ok, buf = cv2.imencode(".jpg", small)
        if not ok:
            print(f"{path}: encode failed")
            continue
        try:
            r = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": VERIFICATION_PROMPT,
                           "images": [buf.tobytes()]}],
                keep_alive="30m",
                options=CPU_OPTIONS,
            )
            raw = r["message"]["content"].strip()
        except Exception as e:
            print(f"{path}: CALL_FAILED {e}")
            continue
        outcome = extract_verification(raw)
        print(f"{path}: box={result['box']} -> verification={outcome!r} (raw={raw!r})")


if __name__ == "__main__":
    main()
