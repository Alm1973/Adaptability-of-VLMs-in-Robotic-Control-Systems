
import cv2
import numpy as np
import ollama

from tracker import Tracker
from colors import RED
from vlm_recovery import MODEL_NAME, OPTIONS
from multimodal_test import (
    count_significant_convexity_defects,
    ellipse_fit_quality,
    specular_highlight_ratio_masked,
)

IMAGES = [
    "obj2_01_home.jpg",
    "obj2_02_verify_base_plus.jpg",
    "obj2_03_verify_tilt_plus.jpg",
    "obj2_04_verify_return_home.jpg",
    "obj2_05_left_mid.jpg",
    "obj2_06_far_left.jpg",
    "obj2_07_left_low.jpg",
    "obj2_08_left_lower.jpg",
    "obj2_09_center_low.jpg",
    "obj2_10_right_low.jpg",
    "obj2_11_far_right.jpg",
    "obj2_12_right_high.jpg",
    "obj2_13_right_higher.jpg",
    "obj2_14_center_high.jpg",
    "index1_test2.jpg",
    "hand_test.jpg",
]

DESCRIBE_PROMPT = (
    "What is the main object visible in this image? Answer with a short "
    "phrase naming the object."
)


CPU_OPTIONS = dict(OPTIONS)
CPU_OPTIONS["num_gpu"] = 0


def qwen_describe(frame):
    small = cv2.resize(frame, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    if not ok:
        return "ENCODE_FAILED"
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": DESCRIBE_PROMPT,
                       "images": [buf.tobytes()]}],
            keep_alive="30m",
            options=CPU_OPTIONS,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"CALL_FAILED: {e}"


def red_contour_metrics(frame):
    tracker = Tracker()
    result = tracker.find_red(frame)
    if result is None:
        return None
    box = result["box"]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in RED:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    for c in contours:
        if cv2.boundingRect(c) == box:
            contour = c
            break
    if contour is None:
        return {"box": box, "note": "contour re-isolation failed"}

    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "box": box,
        "area": area,
        "solidity": round(solidity, 3),
        "circularity": round(circularity, 3),
        "convexity_defects": count_significant_convexity_defects(contour),
        "ellipse_fit": round(ellipse_fit_quality(contour), 3)
                       if ellipse_fit_quality(contour) is not None else None,
        "specular_masked": round(specular_highlight_ratio_masked(frame, contour, box), 4),
    }


def main():
    for path in IMAGES:
        frame = cv2.imread(path)
        print(f"=== {path} ===")
        if frame is None:
            print("  FAILED TO LOAD")
            continue

        answer = qwen_describe(frame)
        print(f"  qwen: {answer!r}")

        metrics = red_contour_metrics(frame)
        if metrics is None:
            print("  red-contour metrics: N/A (no red detection)")
        else:
            print(f"  red-contour metrics: {metrics}")
        print()


if __name__ == "__main__":
    main()
