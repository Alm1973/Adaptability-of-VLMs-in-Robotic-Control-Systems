
import cv2
from camera import Camera

OUTPUT_PATH = "hand_test.jpg"


def main():
    cam = Camera()

    print("Live preview open. Position your hand, then press SPACE or 's' "
          "to capture, or 'q' to quit without saving.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Camera error")
                break

            cv2.imshow("Capture Hand Test - SPACE/s to save, q to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') or key == ord('s'):
                cv2.imwrite(OUTPUT_PATH, frame)
                print(f"Saved {OUTPUT_PATH}")
                break
            elif key == ord('q'):
                print("Quit without saving.")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
