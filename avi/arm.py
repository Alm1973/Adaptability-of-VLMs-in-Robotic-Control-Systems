import serial
import time

class Arm:
    def __init__(self):
        self.serial = serial.Serial('COM5', 115200, timeout=1)
        time.sleep(2)
        self.base = 90
        self.tilt = 90
        self.connection_healthy = True

    def update(self, moveX, moveY):
        if not self.connection_healthy:
            raise RuntimeError(
                "Arm connection marked unhealthy after a prior write "
                "failure -- refusing further relative moves. Call "
                "reconnect() only after independently confirming the link "
                "is actually good again (see post-mortem, point 4/5: this "
                "should be a deliberate human-observed step, not automatic)."
            )

        old_base = self.base
        old_tilt = self.tilt

        new_base = max(0, min(180, old_base - moveX))
        new_tilt = max(0, min(180, old_tilt - moveY))

        actual_moveX = new_base - old_base
        actual_moveY = new_tilt - old_tilt

        if actual_moveX == 0 and actual_moveY == 0:
            return

        msg = f"X:{actual_moveX} Y:{actual_moveY}\n"

        try:
            self.serial.write(msg.encode())
        except Exception:
            self.connection_healthy = False
            raise

        self.base = new_base
        self.tilt = new_tilt
        print("[ARM SENT]", msg.strip())

    def reconnect(self):
        try:
            if self.serial.is_open:
                self.serial.close()
        except Exception:
            pass

        self.serial = serial.Serial('COM5', 115200, timeout=1)
        time.sleep(2)
        self.connection_healthy = True
        print("[ARM] Reconnected. Position tracking is UNVERIFIED -- "
              "confirm actual physical position by eye before trusting "
              "any further relative moves.")