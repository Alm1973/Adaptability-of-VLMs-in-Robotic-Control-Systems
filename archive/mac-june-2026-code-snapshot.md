# Mac `~/avi` folder — June 2026 modular AVI package (code snapshot)

> Source: the author's Mac, folder `~/avi`. Imported 2026-09-28 via the device bridge, verbatim.
> Phase: June 22–30, 2026 (HSV red-cup tracking, SEARCHING/TRACKING/REACQUIRE, Mac serial). Matches lab-notebook entry 2026-06-29 "Modular AVI System + Red Cup Tracking". **Not** the final Sept 1 experiment code (that is on the MSI).
> Also in the folder but not imported: `avi.jpg`, `test.jpg` (1280×720 camera test frames, June 22), `venv/` (Python env, ~165 MB), `__pycache__/`. `config/target.json` is empty.
> Other Mac folders checked: `~/projects` = glucose-ring project (not AVI). Not searched: Desktop, Documents, Downloads (need separate permission).

## `main.py`  (modified 2026-06-30 03:11 UTC, 72 lines)

```python
import cv2
import time

from planner import get_strategy
from vision import detect
from controller import Controller
from state_manager import StateManager

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ctrl = Controller()
sm = StateManager()

strategy = "scan"
color = "red"

last_vlm = 0

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    # -----------------------------
    # DETECTION
    # -----------------------------
    result = detect(frame, color)

    state = sm.update(result)

    # -----------------------------
    # VLM (slow reasoning loop)
    # -----------------------------
    if time.time() - last_vlm > 3:

        strategy, color = get_strategy(frame, state)
        print("STATE:", state, "| STRATEGY:", strategy, "| COLOR:", color)

        last_vlm = time.time()

    # -----------------------------
    # EXECUTION LAYER
    # -----------------------------
    if state == "SEARCHING":
        ctrl.ser.write(b"SCAN\n")

    elif state == "TRACKING":

        if result:
            cx, cy, _ = result
            ctrl.update(cx, "tracking")

    elif state == "LOCKED":
        ctrl.update(None, "lock")

    elif state == "REACQUIRE":
        ctrl.ser.write(b"SCAN_FAST\n")

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.imshow("AVI V2.1 Recovery", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## `tracker.py`  (modified 2026-06-30 02:58 UTC, 164 lines)

```python
import cv2
import numpy as np
import serial
import time
import ollama
from colors import HSV_COLORS

# -----------------------------
# SERIAL
# -----------------------------
ser = serial.Serial("/dev/cu.usbmodem11101", 115200, timeout=1)
time.sleep(2)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FRAME_CENTER = 640

cv2.namedWindow("AVI VLM Tracker", cv2.WINDOW_NORMAL)

# -----------------------------
# STATE
# -----------------------------
TARGET = "red"
last_vlm_time = 0
VLM_INTERVAL = 3.0  # seconds

alpha = 0.85
smooth_cx = None

last_send = 0
SEND_RATE = 0.05

# -----------------------------
# VLM FUNCTION
# -----------------------------
def ask_vlm(frame):
    """
    Ask model which object to track
    """

    _, img = cv2.imencode('.jpg', frame)

    response = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Look at this image. "
                    "Choose ONE object to track from: red, green, blue, yellow. "
                    "Respond with only one word."
                ),
                "images": [img.tobytes()]
            }
        ]
    )

    text = response["message"]["content"].strip().lower()

    for c in ["red", "green", "blue", "yellow"]:
        if c in text:
            return c

    return "red"

# -----------------------------
# SEND
# -----------------------------
def send(error):
    global last_send
    now = time.time()

    if now - last_send > SEND_RATE:
        ser.write((str(error) + "\n").encode())
        last_send = now

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        continue

    # -----------------------------
    # VLM UPDATE (slow loop)
    # -----------------------------
    now = time.time()
    if now - last_vlm_time > VLM_INTERVAL:
        try:
            TARGET = ask_vlm(frame)
            print("VLM TARGET:", TARGET)
        except Exception as e:
            print("VLM error:", e)

        last_vlm_time = now

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = None
    for lower, upper in HSV_COLORS[TARGET]:
        lower = np.array(lower)
        upper = np.array(upper)
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else (mask | m)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    error = 0

    if contours:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 300:

            x, y, w, h = cv2.boundingRect(largest)

            cx_raw = x + w // 2

            if smooth_cx is None:
                smooth_cx = cx_raw
            else:
                smooth_cx = alpha * smooth_cx + (1 - alpha) * cx_raw

            cx = int(smooth_cx)

            error = FRAME_CENTER - cx

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, y + h // 2), 5, (255, 0, 0), -1)

    send(error)

    # -----------------------------
    # UI
    # -----------------------------
    cv2.line(frame, (FRAME_CENTER, 0), (FRAME_CENTER, frame.shape[0]), (255, 0, 0), 2)

    cv2.putText(
        frame,
        f"target: {TARGET} | error: {error}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AVI VLM Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
```

## `state_manager.py`  (modified 2026-06-30 03:09 UTC, 44 lines)

```python
class StateManager:

    def __init__(self):
        self.state = "SEARCHING"
        self.missed_frames = 0
        self.lock_counter = 0

    def update(self, detection):

        # -------------------------
        # OBJECT DETECTED
        # -------------------------
        if detection is not None:

            self.missed_frames = 0

            if self.state == "SEARCHING":
                self.state = "TRACKING"

            elif self.state == "TRACKING":
                self.lock_counter += 1

                if self.lock_counter > 10:
                    self.state = "LOCKED"

            elif self.state == "LOCKED":
                # if drift happens → re-track
                self.state = "TRACKING"

        # -------------------------
        # OBJECT LOST
        # -------------------------
        else:

            self.missed_frames += 1
            self.lock_counter = 0

            if self.missed_frames > 5:
                self.state = "SEARCHING"

            if self.missed_frames > 15:
                self.state = "REACQUIRE"

        return self.state
```

## `planner.py`  (modified 2026-06-30 03:09 UTC, 48 lines)

```python
import ollama
import cv2

def get_strategy(frame, state):

    _, img = cv2.imencode(".jpg", frame)

    prompt = f"""
You are a robotics planner.

Current state: {state}

Return ONLY JSON:
{{
 "strategy": "color_tracking | scan | lock | recover",
 "color": "red | green | blue | yellow | none"
}}

Rules:
- If state is SEARCHING → scan
- If TRACKING → color_tracking
- If LOCKED → lock
- If REACQUIRE → recover
"""

    response = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img.tobytes()]
        }]
    )

    text = response["message"]["content"].lower()

    strategy = "scan"
    color = "red"

    for s in ["color_tracking", "scan", "lock", "recover"]:
        if s in text:
            strategy = s

    for c in ["red", "green", "blue", "yellow"]:
        if c in text:
            color = c

    return strategy, color
```

## `controller.py`  (modified 2026-06-30 03:08 UTC, 25 lines)

```python
import serial
import time

class Controller:

    def __init__(self, port="/dev/cu.usbmodem11101"):
        self.ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)

        self.center = 640  # 1280 width

        self.alpha = 0.7
        self.filtered = 0

    def update(self, cx, mode):

        if mode == "lock" or cx is None:
            return

        error = self.center - cx

        # smoothing (kills jitter)
        self.filtered = self.alpha * self.filtered + (1 - self.alpha) * error

        self.ser.write(f"{int(self.filtered)}\n".encode())
```

## `vision.py`  (modified 2026-06-30 03:07 UTC, 32 lines)

```python
import cv2
import numpy as np
from colors import HSV_COLORS

def detect(frame, color):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = None

    for lower, upper in HSV_COLORS[color]:
        lower = np.array(lower)
        upper = np.array(upper)
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else (mask | m)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 300:
        return None

    x, y, w, h = cv2.boundingRect(largest)

    cx = x + w // 2
    cy = y + h // 2

    return cx, cy, (x, y, w, h)
```

## `colors.py`  (modified 2026-06-29 20:30 UTC, 20 lines)

```python
HSV_COLORS = {

    "red": [
        ((0,120,70),(10,255,255)),
        ((170,120,70),(180,255,255))
    ],

    "green": [
        ((35,60,60),(85,255,255))
    ],

    "blue": [
        ((100,150,50),(140,255,255))
    ],

    "yellow": [
        ((20,100,100),(35,255,255))
    ]

}
```

## `avi_control.py`  (modified 2026-06-22 20:49 UTC, 19 lines)

```python
import serial
import time

SERIAL_PORT = "/dev/cu.usbmodem11101"

ser = serial.Serial(SERIAL_PORT, 115200)
time.sleep(2)

print("AVI CONTROL READY")

while True:
    cmd = input("LEFT / RIGHT / STOP: ").strip().upper()

    if cmd == "EXIT":
        break

    ser.write((cmd + "\n").encode())

ser.close()
```

## `avi_vision.py`  (modified 2026-06-22 21:19 UTC, 65 lines)

```python
import cv2
import serial
import time
import numpy as np

ser = serial.Serial("/dev/cu.usbmodem11101", 115200)
time.sleep(2)

cap = cv2.VideoCapture(0)

FRAME_CENTER = 320  # adjust if needed (640 width camera)

def get_direction(cx):
    error = cx - FRAME_CENTER

    if abs(error) < 30:
        return "STOP"

    if error < 0:
        return "LEFT"
    else:
        return "RIGHT"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red mask
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        cx = x + w // 2

        cmd = get_direction(cx)

        ser.write((cmd + "\n").encode())

        cv2.circle(frame, (cx, y + h // 2), 10, (0, 255, 0), -1)

    cv2.line(frame, (FRAME_CENTER, 0), (FRAME_CENTER, 480), (255, 0, 0), 2)

    cv2.imshow("AVI TRACKING", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
```

## `avi_vlm.py`  (modified 2026-06-22 20:19 UTC, 50 lines)

```python
import cv2
import base64
import requests
import time

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

def encode(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def ask_vlm(image_b64):
    prompt = """
You are controlling a robot.

Task: Find a red cup.

Return ONLY one word:
FOUND
NOT_FOUND
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5vl:3b",
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
    )

    return res.json()["response"].strip()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = encode(frame)

    result = ask_vlm(img)
    print("VLM:", result)

    if "FOUND" in result:
        print("RED CUP FOUND")
        break

    time.sleep(1)
```

## `config/target.json`  (modified 2026-06-29 20:29 UTC, 0 lines)

```json
(empty file)
```
