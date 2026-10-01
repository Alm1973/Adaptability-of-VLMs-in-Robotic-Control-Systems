
import math
import threading
import cv2
import numpy as np

DEFAULT_BUCKET_SIZE_DEG = 30

DEFAULT_ANGLE_MIN = 0
DEFAULT_ANGLE_MAX = 180

DEFAULT_TILE_SIZE = (160, 120)

PLACEHOLDER_GRAY = 60


class TileMap:
    def __init__(self, bucket_size_deg=DEFAULT_BUCKET_SIZE_DEG,
                 angle_min=DEFAULT_ANGLE_MIN, angle_max=DEFAULT_ANGLE_MAX,
                 tile_size=DEFAULT_TILE_SIZE):
        self.bucket_size_deg = bucket_size_deg
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.tile_size = tile_size

        self.num_buckets = math.ceil((angle_max - angle_min) / bucket_size_deg)

        self.tiles = {}

        self._lock = threading.Lock()

    def _bucket(self, angle):
        clamped = max(self.angle_min, min(self.angle_max, angle))
        idx = int((clamped - self.angle_min) // self.bucket_size_deg)
        return min(idx, self.num_buckets - 1)

    def update(self, frame, base_angle, tilt_angle):
        base_bucket = self._bucket(base_angle)
        tilt_bucket = self._bucket(tilt_angle)

        resized = cv2.resize(frame, self.tile_size)
        with self._lock:
            self.tiles[(base_bucket, tilt_bucket)] = resized

        return (base_bucket, tilt_bucket)

    def coverage(self):
        with self._lock:
            filled = len(self.tiles)
        total = self.num_buckets * self.num_buckets
        return filled / total if total else 0.0

    def build_mosaic(self):
        with self._lock:
            tiles_snapshot = dict(self.tiles)

        tile_w, tile_h = self.tile_size
        canvas = np.full(
            (self.num_buckets * tile_h, self.num_buckets * tile_w, 3),
            PLACEHOLDER_GRAY,
            dtype=np.uint8,
        )

        for (base_bucket, tilt_bucket), tile in tiles_snapshot.items():
            x0 = base_bucket * tile_w
            y0 = tilt_bucket * tile_h
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        return canvas
