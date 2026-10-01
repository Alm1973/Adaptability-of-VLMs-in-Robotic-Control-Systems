import time
from collections import deque

from config import (
    POSITION_HISTORY_LEN, MIN_DIRECTION_HISTORY, MIN_DIRECTION_MOVEMENT_PX,
    SUPPRESSION_RADIUS_PX, SUPPRESSION_ZONE_EXPIRY_SEC,
)

DEFAULT_MODEL = "yolov8s-worldv2.pt"

MIN_CONFIDENCE = 0.25

CONTEXT_CLASSES = ["laptop", "keyboard", "computer mouse", "water bottle",
                   "computer monitor", "soda can", "headphones",
                   "banana", "bicycle", "elephant", "umbrella"]

MIN_CONSECUTIVE_HITS = 2

HOLD_CONFIDENCE = 0.15

UNRELIABLE_CLASSES = {
    "water bottle": "inverted confidences on held-out data (a can outranks "
                    "every real bottle); use open_vocab_detect.detect_multiview",
    "bottle": "see 'water bottle'",
}


class YoloTracker:

    DEBUG_FLOOR = 0.01

    def __init__(self, model_name=DEFAULT_MODEL, min_confidence=MIN_CONFIDENCE,
                 default_target="computer mouse", debug=False,
                 hold_confidence=HOLD_CONFIDENCE):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(model_name)
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.hold_confidence = min(hold_confidence, min_confidence)
        self.default_target = default_target
        self.debug = debug
        self.debug_log = []

        self._classes = None
        self._targets = set()
        self._streak_label = None
        self._streak = 0

        self.position_history = deque(maxlen=POSITION_HISTORY_LEN)
        self.suppression_zones = []


    def add_suppression_zone(self, center):
        cx, cy = center
        self.suppression_zones.append((cx, cy, time.time()))
        print(f"[YOLO] Added suppression zone at ({cx}, {cy}), "
              f"active zones: {len(self.suppression_zones)}")

    def _purge_expired_suppression_zones(self):
        now = time.time()
        before = len(self.suppression_zones)
        self.suppression_zones = [
            z for z in self.suppression_zones
            if now - z[2] < SUPPRESSION_ZONE_EXPIRY_SEC
        ]
        if before - len(self.suppression_zones):
            print(f"[YOLO] {before - len(self.suppression_zones)} "
                  f"suppression zone(s) expired")

    def _is_suppressed(self, cx, cy):
        return any(
            ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5 <= SUPPRESSION_RADIUS_PX
            for zx, zy, _ in self.suppression_zones
        )

    def reset_history(self):
        self.position_history.clear()
        self._streak_label = None
        self._streak = 0

    def get_last_known_direction(self):
        if len(self.position_history) < MIN_DIRECTION_HISTORY:
            print(f"[YOLO] Not enough history for direction "
                  f"({len(self.position_history)} frames)")
            return None
        x0, y0 = self.position_history[0]
        x1, y1 = self.position_history[-1]
        dx, dy = x1 - x0, y1 - y0
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < MIN_DIRECTION_MOVEMENT_PX:
            print(f"[YOLO] Object roughly stationary before loss "
                  f"(moved {distance:.1f}px) -> no direction hint")
            return None
        if abs(dx) >= abs(dy):
            direction = "scan_right" if dx > 0 else "scan_left"
        else:
            direction = "tilt_down" if dy > 0 else "tilt_up"
        print(f"[YOLO] Last known direction before loss: {direction} "
              f"(dx={dx}, dy={dy}, frames={len(self.position_history)})")
        return direction


    def set_targets(self, labels, allow_unreliable=False, quiet=False):
        targets = list(labels)
        bad = [l for l in targets if l in UNRELIABLE_CLASSES]
        if bad and not allow_unreliable:
            raise ValueError(
                "Refusing to track measured-unreliable class(es) "
                + ", ".join(f"{b!r} ({UNRELIABLE_CLASSES[b]})" for b in bad)
                + " -- pass allow_unreliable=True to override"
            )
        if bad and not quiet:
            for b in bad:
                print(f"[YOLO] WARNING tracking UNRELIABLE class {b!r}: "
                      f"{UNRELIABLE_CLASSES[b]}")
        full = targets + [c for c in CONTEXT_CLASSES if c not in targets]
        if full == self._classes:
            self._targets = set(targets)
            return
        self.model.set_classes(full)
        self._classes = full
        self._targets = set(targets)
        self._streak_label, self._streak = None, 0
        if not quiet:
            print(f"[YOLO] targets {targets} "
                  f"(+{len(full) - len(targets)} context)")

    def find(self, frame, target=None):
        self._purge_expired_suppression_zones()

        if target is not None:
            self.set_targets([target] if isinstance(target, str) else target)
        if not self._classes:
            raise RuntimeError("call set_targets() before find()")

        locked = self._streak >= MIN_CONSECUTIVE_HITS
        active_threshold = self.hold_confidence if locked else self.min_confidence

        floor = min(active_threshold, self.DEBUG_FLOOR) if self.debug \
            else active_threshold
        res = self.model.predict(frame, conf=floor, verbose=False)[0]

        if self.debug:
            raw = sorted(
                ((self.model.names[int(b.cls.item())], float(b.conf.item()))
                 for b in res.boxes), key=lambda x: -x[1])
            tgt = [(n, c) for n, c in raw if n in self._targets]
            self.debug_log.append({
                "target_raw": [(n, round(c, 3)) for n, c in tgt],
                "best_raw": round(tgt[0][1], 3) if tgt else 0.0,
                "over_threshold": bool(tgt and tgt[0][1] >= self.min_confidence),
                "streak_before": self._streak,
            })

        best = None
        for b in res.boxes:
            label = self.model.names[int(b.cls.item())]
            if label not in self._targets:
                continue
            conf = float(b.conf.item())
            if conf < active_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if self._is_suppressed(cx, cy):
                continue
            if best is None or conf > best["conf"]:
                best = {
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "center": (cx, cy),
                    "area": (x2 - x1) * (y2 - y1),
                    "conf": round(conf, 3),
                    "label": label,
                }

        if best is None:
            self._streak_label, self._streak = None, 0
            return None
        if best["label"] != self._streak_label:
            self._streak_label, self._streak = best["label"], 1
        else:
            self._streak += 1
        if self._streak < MIN_CONSECUTIVE_HITS:
            return None

        self.position_history.append(best["center"])
        return best

    def find_red(self, frame):
        if not self._classes:
            self.set_targets([self.default_target])
        return self.find(frame)
