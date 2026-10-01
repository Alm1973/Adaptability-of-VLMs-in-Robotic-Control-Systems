

class Controller:

    def __init__(self):
        self.kp = 0.015
        self.deadzone = 80
        self.max_step = 3

        self.smoothing = 0.35
        self.smooth_span = 4.0
        self.slew = 1

        self._ex = self._ey = 0.0
        self._rx = self._ry = 0.0
        self._px = self._py = 0
        self._primed = False

    def reset(self):
        self._ex = self._ey = 0.0
        self._rx = self._ry = 0.0
        self._px = self._py = 0
        self._primed = False

    def _alpha(self, err):
        span = self.smooth_span * self.deadzone
        frac = min(1.0, abs(err) / span) if span > 0 else 1.0
        return self.smoothing + (1.0 - self.smoothing) * frac

    def _axis(self, err, smoothed, residual, prev):
        a = self._alpha(err)
        s = float(err) if not self._primed else (
            a * err + (1.0 - a) * smoothed)

        if s > self.deadzone:
            effective = s - self.deadzone
        elif s < -self.deadzone:
            effective = s + self.deadzone
        else:
            effective = 0.0
            residual = 0.0

        want = self.kp * effective + residual
        cmd = int(want)
        residual = want - cmd

        if cmd > self.max_step:
            cmd = self.max_step
        elif cmd < -self.max_step:
            cmd = -self.max_step

        if cmd * prev >= 0 and abs(cmd) > abs(prev) + self.slew:
            step = abs(prev) + self.slew
            cmd = step if cmd > 0 else -step

        if abs(residual) > 1.0:
            residual = 0.0
        return cmd, s, residual

    def compute(self, errorX, errorY):
        cx, self._ex, self._rx = self._axis(errorX, self._ex, self._rx,
                                            self._px)
        cy, self._ey, self._ry = self._axis(errorY, self._ey, self._ry,
                                            self._py)
        self._primed = True
        self._px, self._py = cx, cy
        return cx, cy
