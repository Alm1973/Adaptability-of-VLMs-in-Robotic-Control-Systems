
import cv2
import ollama
import time
import argparse
import sys

CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

RECOVERY_PROMPT = (
    "You are controlling a camera-mounted robotic arm that has lost track "
    "of a red cup. Look at this image and respond with exactly one action "
    "from this list: scan_left, scan_right, tilt_up, tilt_down, object_found. "
    "Respond with only the action word, nothing else."
)


def capture_frame(resize=None):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Camera failed to open. Check CAMERA_INDEX.")
        sys.exit(1)

    for _ in range(5):
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        print("ERROR: Failed to capture frame.")
        sys.exit(1)

    if resize:
        w, h = resize
        frame = cv2.resize(frame, (w, h))

    return frame


def run_vlm_test(model_name, resize=None):
    print(f"\n=== Testing model: {model_name} | resize: {resize or 'none (full res)'} ===")

    t0 = time.time()
    frame = capture_frame(resize=resize)
    t1 = time.time()
    capture_time = t1 - t0

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print("ERROR: Failed to encode frame.")
        sys.exit(1)
    image_bytes = buffer.tobytes()
    t2 = time.time()
    encode_time = t2 - t1

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": RECOVERY_PROMPT,
                "images": [image_bytes],
            }
        ],
    )
    t3 = time.time()
    inference_time = t3 - t2

    total_time = t3 - t0
    action_text = response["message"]["content"].strip()

    print(f"Capture time:    {capture_time:.3f}s")
    print(f"Encode time:     {encode_time:.3f}s")
    print(f"Inference time:  {inference_time:.3f}s")
    print(f"TOTAL time:      {total_time:.3f}s")
    print(f"Image size:      {frame.shape[1]}x{frame.shape[0]}")
    print(f"Model response:  {action_text}")

    return {
        "model": model_name,
        "resize": resize,
        "capture_time": capture_time,
        "encode_time": encode_time,
        "inference_time": inference_time,
        "total_time": total_time,
        "response": action_text,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model name")
    parser.add_argument("--resize", default=None, help="Optional WxH, e.g. 320x240")
    args = parser.parse_args()

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    run_vlm_test(args.model, resize=resize)