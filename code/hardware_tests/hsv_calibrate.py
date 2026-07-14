"""
hsv_calibrate.py — interactive HSV range finder

Run this with your actual object in front of the actual camera, under your
actual lighting. Drag the six sliders until only your object shows up white
in the "mask" window (everything else black). Whatever numbers the sliders
land on are your real calibrated values — copy them into detection.py's
COLOR_RANGES for that color.

Usage:
    python hsv_calibrate.py
Press ESC to quit and print the final values.
"""

import cv2
import numpy as np

CAM_INDEX = 0


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow("mask")

    # Start with wide-open bounds; narrow them with the sliders.
    cv2.createTrackbar("H min", "mask", 0, 180, nothing)
    cv2.createTrackbar("H max", "mask", 180, 180, nothing)
    cv2.createTrackbar("S min", "mask", 0, 255, nothing)
    cv2.createTrackbar("S max", "mask", 255, 255, nothing)
    cv2.createTrackbar("V min", "mask", 0, 255, nothing)
    cv2.createTrackbar("V max", "mask", 255, 255, nothing)

    print("Hold your object in frame. Drag sliders until only the object "
          "is white in the mask window. Press ESC when done.")

    h_min = h_max = s_min = s_max = v_min = v_max = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", "mask")
        h_max = cv2.getTrackbarPos("H max", "mask")
        s_min = cv2.getTrackbarPos("S min", "mask")
        s_max = cv2.getTrackbarPos("S max", "mask")
        v_min = cv2.getTrackbarPos("V min", "mask")
        v_max = cv2.getTrackbarPos("V max", "mask")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal values — copy into detection.py COLOR_RANGES:")
    print(f'    (np.array([{h_min}, {s_min}, {v_min}]), '
          f'np.array([{h_max}, {s_max}, {v_max}])),')
    print("\nNote: red wraps around hue 0/180 — if your red object needs "
          "two ranges, run this twice (once near hue 0, once near hue 180) "
          "and keep both, same as the existing red entry in COLOR_RANGES.")


if __name__ == "__main__":
    main()
