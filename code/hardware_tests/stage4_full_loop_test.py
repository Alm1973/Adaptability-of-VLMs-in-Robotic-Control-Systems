"""

"""

import sys
import time

import cv2
import serial

from detection import classify_grid, detect_any

PORT = "/dev/cu.usbmodem1101"   # TODO(Shaurya): confirm your port
BAUD = 115200
CAM_INDEX = 0


HARDCODE_ACTION = None

ACTION_VOCAB = [
    "ROTATE_BASE_LEFT_15",
    "ROTATE_BASE_RIGHT_15",
    "TILT_UP_10",
    "TILT_DOWN_10",
    "CONFIRM_TARGET",
]

PROMPT_TEMPLATE = """
An object was detected at grid position: {grid_label}.
Frame center is "center". Choose exactly ONE action from this list:
ROTATE_BASE_LEFT_15
ROTATE_BASE_RIGHT_15
TILT_UP_10
TILT_DOWN_10
CONFIRM_TARGET

Rules:
- left-side grid positions -> ROTATE_BASE_LEFT_15
- right-side grid positions -> ROTATE_BASE_RIGHT_15
- top positions -> TILT_UP_10
- bottom positions -> TILT_DOWN_10
- "center" -> CONFIRM_TARGET
Return ONLY the action, nothing else.
"""


def query_vlm(grid_label):
    """Ask Moondream (via Ollama) for an action given a grid label."""
    import ollama

    response = ollama.chat(
        model="moondream",
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(grid_label=grid_label)}],
    )
    text = response["message"]["content"].strip()

    for action in ACTION_VOCAB:
        if action in text:
            return action
    print(f"WARNING: VLM returned unrecognized text: {text!r}")
    return "CONFIRM_TARGET"  # safe fallback


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAM_INDEX}.")
        sys.exit(1)

    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    while arduino.in_waiting:
        arduino.readline()  # drain READY banner

    print("Full loop test running. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        result = detect_any(frame)

        if result:
            cx, cy = result["centroid"]
            grid_label = classify_grid((cx, cy), w, h)

            t0 = time.time()
            if HARDCODE_ACTION:
                action = HARDCODE_ACTION
            else:
                action = query_vlm(grid_label)
            decide_time = time.time() - t0

            arduino.write((action + "\n").encode())
            time.sleep(0.1)
            while arduino.in_waiting:
                print("Arduino:", arduino.readline().decode(errors="ignore").strip())

            print(f"Detected: {grid_label} | Action: {action} | "
                  f"Decide time: {decide_time:.2f}s")

            x, y, bw, bh = result["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        cv2.imshow("Stage 4: Full Loop Test", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()


if __name__ == "__main__":
    main()


