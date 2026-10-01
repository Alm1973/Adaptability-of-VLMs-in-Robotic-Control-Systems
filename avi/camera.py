import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise Exception("Camera failed to open")

        self.warm = self._warm_up()

    WARMUP_TIMEOUT_S = 8.0
    BLANK_MEAN_THRESHOLD = 1.0

    def _warm_up(self):
        import time
        t0 = time.time()
        reads = 0
        while time.time() - t0 < self.WARMUP_TIMEOUT_S:
            ok, frame = self.cap.read()
            reads += 1
            if ok and frame is not None and frame.mean() >= self.BLANK_MEAN_THRESHOLD:
                print(f"[CAMERA] warm after {reads} reads / "
                      f"{time.time() - t0:.1f}s")
                return True
            time.sleep(0.15)
        print(f"[CAMERA] WARNING: still blank after {reads} reads / "
              f"{self.WARMUP_TIMEOUT_S}s. Frames may be black -- do NOT drive "
              f"hardware on them (check self.warm).")
        return False

    def read(self):
        try:
            return self.cap.read()
        except Exception as e:
            print(f"[CAMERA] Read failed: {e}")
            return False, None

    def read_fresh(self, drain=2):
        try:
            for _ in range(max(0, drain)):
                self.cap.grab()
            return self.cap.read()
        except Exception as e:
            print(f"[CAMERA] Fresh read failed: {e}")
            return False, None

    def release(self):
        self.cap.release()