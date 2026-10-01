
import cv2
import numpy as np
import math

from tracker import Tracker

TEST_IMAGES = ["index1_test2.jpg", "hand_test.jpg"]


def count_significant_convexity_defects(contour, depth_threshold_px=20):
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if len(hull_indices) < 4:
        return 0
    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    defects = defects.reshape(-1, 4)
    count = 0
    for i in range(defects.shape[0]):
        d = defects[i, 3]
        depth_px = d / 256.0
        if depth_px > depth_threshold_px:
            count += 1
    return count


def ellipse_fit_quality(contour):
    if len(contour) < 5:
        return None
    (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
    ellipse_area = math.pi * (major / 2) * (minor / 2)
    if ellipse_area == 0:
        return None
    contour_area = cv2.contourArea(contour)
    return contour_area / ellipse_area


def hough_circle_check(frame, box):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    min_dim = min(w, h)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim / 2,
        param1=50, param2=30,
        minRadius=int(min_dim * 0.2), maxRadius=int(min_dim * 0.6),
    )
    return 0 if circles is None else circles.shape[1]


def specular_highlight_ratio(frame, box):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    highlight_mask = (v > 220) & (s < 60)
    return float(highlight_mask.sum()) / highlight_mask.size


def specular_highlight_ratio_masked(frame, contour, box):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - [x, y]
    cv2.drawContours(contour_mask, [shifted], -1, 255, thickness=cv2.FILLED)
    in_contour = contour_mask > 0

    highlight_mask = (v > 220) & (s < 60) & in_contour
    contour_pixel_count = in_contour.sum()
    if contour_pixel_count == 0:
        return 0.0
    return float(highlight_mask.sum()) / contour_pixel_count


def run(path):
    print(f"=== {path} ===")
    frame = cv2.imread(path)
    if frame is None:
        print("  FAILED TO LOAD IMAGE")
        return

    tracker = Tracker()
    result = tracker.find_red(frame)
    if result is None:
        print("  No red-ish contour found")
        return

    box = result["box"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    from colors import RED
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
        print("  Could not re-isolate contour for defect/ellipse analysis")
        return

    defect_count = count_significant_convexity_defects(contour)
    ellipse_quality = ellipse_fit_quality(contour)
    hough_circles_found = hough_circle_check(frame, box)
    highlight_ratio_box = specular_highlight_ratio(frame, box)
    highlight_ratio_masked = specular_highlight_ratio_masked(frame, contour, box)

    print(f"  Box:                        {box}")
    print(f"  Convexity defects (>20px):  {defect_count}")
    print(f"  Ellipse fit quality:        {ellipse_quality:.3f}" if ellipse_quality is not None else "  Ellipse fit quality:        N/A")
    print(f"  Hough circles found:        {hough_circles_found}")
    print(f"  Specular highlight ratio (box):    {highlight_ratio_box:.4f}")
    print(f"  Specular highlight ratio (masked): {highlight_ratio_masked:.4f}")


if __name__ == "__main__":
    for path in TEST_IMAGES:
        run(path)
        print()
