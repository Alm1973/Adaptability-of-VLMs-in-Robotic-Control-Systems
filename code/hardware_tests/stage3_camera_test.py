"""

Prints FPS and detection status every frame
"""

import time

import cv2

from detection import classify_grid, detect_any

CAM_INDEX = 0  # confirm with `ls /dev/video*` or trial and error


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAM_INDEX}. "
              f"Try a different index (0 vs 1 is a common mixup).")
        return

    print("Camera open. Press ESC to quit.")
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            continue

        frame_count += 1
        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0

        h, w = frame.shape[:2]
        result = detect_any(frame)

        status = "NO DETECTION"
        if result:
            x, y, bw, bh = result["bbox"]
            cx, cy = result["centroid"]
            label = classify_grid((cx, cy), w, h)
            status = f"{result['color']} @ {label}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Stage 3: Camera Test", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final average FPS: {fps:.1f}")


if __name__ == "__main__":
    main()


