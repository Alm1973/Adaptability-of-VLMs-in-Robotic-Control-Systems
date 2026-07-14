"""

"""

import cv2
import numpy as np


COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
    "blue": [
        (np.array([100, 100, 70]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40, 70, 70]), np.array([85, 255, 255])),
    ],
}

MIN_CONTOUR_AREA = 700  # filter out noise
MORPH_KERNEL = np.ones((5, 5), np.uint8)


def build_mask(hsv_frame, color):
    """OR together all HSV ranges for a color (handles red's hue wraparound)."""
    mask = None
    for lower, upper in COLOR_RANGES[color]:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else (mask + m)
    return mask


def clean_mask(mask):
    """
    Assignment 4(a): morphological cleanup.
    Erosion first to knock out small speckle noise, then dilation to fill
    back in any small gaps left inside the real object blob.
    """
    eroded = cv2.erode(mask, MORPH_KERNEL, iterations=1)
    dilated = cv2.dilate(eroded, MORPH_KERNEL, iterations=2)
    return dilated


def detect_object(frame, color):
    """
    Full pipeline: HSV mask -> erosion -> dilation -> find contours ->
    filter by area -> centroid.

    Returns a dict with bounding box + centroid, or None if nothing found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color)
    mask = clean_mask(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2

    return {
        "color": color,
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "area": cv2.contourArea(largest),
    }


def detect_any(frame, colors=("red", "yellow", "blue", "green")):
    """Try each color and return the first (largest-area) match found."""
    best = None
    for color in colors:
        result = detect_object(frame, color)
        if result and (best is None or result["area"] > best["area"]):
            best = result
    return best



GRID_LABELS = [
    ["top-left", "top-center", "top-right"],
    ["middle-left", "center", "middle-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]


def classify_grid(centroid, frame_width, frame_height):
    cx, cy = centroid
    col = min(int(cx / frame_width * 3), 2)
    row = min(int(cy / frame_height * 3), 2)
    return GRID_LABELS[row][col]


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        result = detect_any(frame)

        if result:
            x, y, bw, bh = result["bbox"]
            cx, cy = result["centroid"]
            label = classify_grid((cx, cy), w, h)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"{result['color']} | {label}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Detection (Assignment 4)", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
