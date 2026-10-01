
import sys
import math
import cv2
import numpy as np

from colors import RED


def largest_red_contour(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = None
    for lower, upper in RED:
        current = cv2.inRange(hsv, lower, upper)
        mask = current if mask is None else cv2.bitwise_or(mask, current)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def compute_shape_metrics(contour):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity,
    }


def run(image_path):
    print(f"=== {image_path} ===")

    frame = cv2.imread(image_path)
    if frame is None:
        print("  FAILED TO LOAD IMAGE")
        return

    contour = largest_red_contour(frame)
    if contour is None:
        print("  No red-ish contour found (mask produced zero contours)")
        return

    metrics = compute_shape_metrics(contour)
    print(f"  Area:               {metrics['area']:.1f}")
    print(f"  Aspect ratio (w/h): {metrics['aspect_ratio']:.3f}")
    print(f"  Solidity:           {metrics['solidity']:.3f}")
    print(f"  Circularity:        {metrics['circularity']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shape_test.py image1.jpg [image2.jpg ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        run(path)
        print()
