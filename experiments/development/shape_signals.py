
import cv2

from config import GATE_REJECT_MIN_DEFECTS, GATE_REJECT_MAX_SOLIDITY


def count_significant_convexity_defects(contour, depth_threshold_px=20):
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return 0
    import numpy as np
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
        if defects[i, 3] / 256.0 > depth_threshold_px:
            count += 1
    return count


def compute_shape_signals(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    return {
        "area": area,
        "solidity": round(solidity, 3),
        "convexity_defects": count_significant_convexity_defects(contour),
    }


def classify_from_signals(signals):
    if signals["convexity_defects"] >= GATE_REJECT_MIN_DEFECTS:
        return "other"
    if signals["solidity"] < GATE_REJECT_MAX_SOLIDITY:
        return "other"
    return None
