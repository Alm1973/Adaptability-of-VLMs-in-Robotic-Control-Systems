
import cv2
import ollama
import time
import argparse
import csv
import statistics
from datetime import datetime

CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

RECOVERY_PROMPT = (
    "You are controlling a camera-mounted robotic arm that has lost track "
    "of a red cup. Look at this image and respond with exactly one action "
    "from this list: scan_left, scan_right, tilt_up, tilt_down, object_found. "
    "Respond with only the action word, nothing else."
)

VALID_ACTIONS = {"scan_left", "scan_right", "tilt_up", "tilt_down", "object_found"}

CONFIGS = [
    ("qwen2.5vl:3b", None),
    ("qwen2.5vl:3b", (320, 240)),
    ("qwen2.5vl:3b", (640, 480)),
    ("moondream", None),
    ("moondream", (320, 240)),
]


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Camera failed to open. Check CAMERA_INDEX.")

    for _ in range(5):
        cap.read()

    return cap


def capture_frame(cap, resize=None):
    ret, frame = cap.read()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame.")

    if resize:
        w, h = resize
        frame = cv2.resize(frame, (w, h))

    return frame


def run_single_trial(cap, model_name, resize=None):
    t0 = time.time()
    frame = capture_frame(cap, resize=resize)
    t1 = time.time()

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame.")
    image_bytes = buffer.tobytes()
    t2 = time.time()

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": RECOVERY_PROMPT, "images": [image_bytes]}
        ],
        keep_alive="30m",
        options={"num_ctx": 2048},
    )
    t3 = time.time()

    action_text = response["message"]["content"].strip()
    usable = action_text.lower() in VALID_ACTIONS

    return {
        "capture_time": t1 - t0,
        "encode_time": t2 - t1,
        "inference_time": t3 - t2,
        "total_time": t3 - t0,
        "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "response": action_text,
        "usable": usable,
    }


def main(trials):
    all_results = []

    print("Opening camera once for the whole run...")
    cap = open_camera()

    try:
        for model_name, resize in CONFIGS:
            res_label = f"{resize[0]}x{resize[1]}" if resize else "full (1280x720)"
            print(f"\n=== {model_name} | {res_label} | {trials} trials ===")

            print("  (warming up model with an image call...)")
            warm_frame = capture_frame(cap, resize=resize)
            ok, warm_buf = cv2.imencode(".jpg", warm_frame)
            warmup_start = time.time()
            try:
                ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": "Describe this image in one word.",
                         "images": [warm_buf.tobytes()]}
                    ],
                    keep_alive="30m",
                    options={"num_ctx": 2048},
                )
                print(f"  warm-up took {time.time() - warmup_start:.2f}s (not counted in results)")
            except Exception as e:
                print(f"  warm-up failed: {e}")

            totals = []
            for i in range(trials):
                try:
                    result = run_single_trial(cap, model_name, resize=resize)
                    totals.append(result["total_time"])
                    print(
                        f"  trial {i+1}: total={result['total_time']:.2f}s "
                        f"(capture={result['capture_time']:.2f}s, "
                        f"inference={result['inference_time']:.2f}s) "
                        f"-> {result['response']} "
                        f"[{'OK' if result['usable'] else 'INVALID'}]"
                    )
                    result["model"] = model_name
                    result["resize_setting"] = res_label
                    result["trial_num"] = i + 1
                    all_results.append(result)
                except Exception as e:
                    print(f"  trial {i+1}: FAILED - {e}")

            if totals:
                print(
                    f"  --> avg={statistics.mean(totals):.2f}s "
                    f"min={min(totals):.2f}s max={max(totals):.2f}s"
                )
    finally:
        cap.release()
        print("\nCamera released.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "resize_setting", "trial_num", "capture_time",
                "encode_time", "inference_time", "total_time",
                "image_size", "response", "usable",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Raw results saved to: {csv_filename}")

    print("\n=== SUMMARY TABLE ===")
    print("| Model | Resolution | Avg latency (s) | Min | Max | Usable outputs |")
    print("|---|---|---|---|---|---|")

    seen = set()
    for model_name, resize in CONFIGS:
        res_label = f"{resize[0]}x{resize[1]}" if resize else "1280x720"
        key = (model_name, res_label)
        if key in seen:
            continue
        seen.add(key)

        matching = [
            r for r in all_results
            if r["model"] == model_name and (
                (resize is None and r["resize_setting"] == "full (1280x720)") or
                (resize is not None and r["resize_setting"] == f"{resize[0]}x{resize[1]}")
            )
        ]

        if not matching:
            continue

        times = [r["total_time"] for r in matching]
        usable_count = sum(1 for r in matching if r["usable"])
        print(
            f"| {model_name} | {res_label} | {statistics.mean(times):.2f} | "
            f"{min(times):.2f} | {max(times):.2f} | {usable_count}/{len(matching)} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="Trials per configuration")
    args = parser.parse_args()
    main(args.trials)