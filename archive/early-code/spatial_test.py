import cv2
import numpy as np

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
    return mask

def get_pos(mask, frame):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x = coords[:,0,0].mean()
    y = coords[:,0,1].mean()

    h, w, _ = frame.shape
    return x/w, y/h

def classify(x, y):
    if x < 0.33:
        h = "LEFT"
    elif x > 0.66:
        h = "RIGHT"
    else:
        h = "CENTER"

    if y < 0.33:
        v = "TOP"
    elif y > 0.66:
        v = "BOTTOM"
    else:
        v = "MIDDLE"

    return v + "-" + h


def run_test(name):
    cap = cv2.VideoCapture(0)

    print("\n====================")
    print("SCENARIO:", name)

    for i in range(20):  # average 20 frames for stability
        ret, frame = cap.read()
        if not ret:
            continue

        mask = detect_red(frame)
        pos = get_pos(mask, frame)

        if pos:
            x, y = pos
            label = classify(x, y)
            print("Frame", i, "→", label)

    cap.release()


# RUN ALL SCENARIOS ONE BY ONE
input("Place cup CENTER and press ENTER")
run_test("CENTER")

input("Place cup TOP-LEFT and press ENTER")
run_test("TOP-LEFT")

input("Place cup BOTTOM-RIGHT and press ENTER")
run_test("BOTTOM-RIGHT")

print("\nDONE")