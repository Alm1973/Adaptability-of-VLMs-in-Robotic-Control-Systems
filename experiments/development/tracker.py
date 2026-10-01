import time
import cv2
import numpy as np
from collections import deque

from colors import RED
from config import (
    MIN_OBJECT_AREA, SHOW_MASK,
    POSITION_HISTORY_LEN, MIN_DIRECTION_HISTORY, MIN_DIRECTION_MOVEMENT_PX,
    SUPPRESSION_RADIUS_PX, SUPPRESSION_ZONE_EXPIRY_SEC,
)

class Tracker:

    def __init__(self):
        self.position_history = deque(maxlen=POSITION_HISTORY_LEN)
        self.suppression_zones = []

    def add_suppression_zone(self, center):
        cx, cy = center
        self.suppression_zones.append((cx, cy, time.time()))
        print(f"[TRACKER] Added suppression zone at ({cx}, {cy}), "
              f"active zones: {len(self.suppression_zones)}")

    def _purge_expired_suppression_zones(self):
        now = time.time()
        before = len(self.suppression_zones)
        self.suppression_zones = [
            z for z in self.suppression_zones
            if now - z[2] < SUPPRESSION_ZONE_EXPIRY_SEC
        ]
        removed = before - len(self.suppression_zones)
        if removed:
            print(f"[TRACKER] {removed} suppression zone(s) expired")

    def _is_suppressed(self, cx, cy):
        return any(
            ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5 <= SUPPRESSION_RADIUS_PX
            for zx, zy, _ in self.suppression_zones
        )

    def find_red(self, frame):

        self._purge_expired_suppression_zones()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = None

        for lower, upper in RED:
            current = cv2.inRange(hsv, lower, upper)

            if mask is None:
                mask = current
            else:
                mask = cv2.bitwise_or(mask, current)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if SHOW_MASK:
            cv2.imshow("mask", mask)

        if not contours:
            return None

        candidates = sorted(contours, key=cv2.contourArea, reverse=True)

        for candidate in candidates:
            area = cv2.contourArea(candidate)
            if area < MIN_OBJECT_AREA:
                break

            x, y, w, h = cv2.boundingRect(candidate)
            cx = x + w // 2
            cy = y + h // 2

            if self._is_suppressed(cx, cy):
                continue

            self.position_history.append((cx, cy))

            return {
                "box": (x, y, w, h),
                "center": (cx, cy),
                "area": area,
                "contour": candidate
            }

        return None

    def reset_history(self):
        self.position_history.clear()

    def get_last_known_direction(self):
        if len(self.position_history) < MIN_DIRECTION_HISTORY:
            print(f"[TRACKER] Not enough history for direction "
                  f"({len(self.position_history)} frames)")
            return None

        x0, y0 = self.position_history[0]
        x1, y1 = self.position_history[-1]
        dx = x1 - x0
        dy = y1 - y0
        distance = (dx * dx + dy * dy) ** 0.5

        if distance < MIN_DIRECTION_MOVEMENT_PX:
            print(f"[TRACKER] Object roughly stationary before loss "
                  f"(moved {distance:.1f}px) -> no direction hint")
            return None

        if abs(dx) >= abs(dy):
            direction = "scan_right" if dx > 0 else "scan_left"
        else:
            direction = "tilt_down" if dy > 0 else "tilt_up"

        print(f"[TRACKER] Last known direction before loss: {direction} "
              f"(dx={dx}, dy={dy}, frames={len(self.position_history)})")
        return direction