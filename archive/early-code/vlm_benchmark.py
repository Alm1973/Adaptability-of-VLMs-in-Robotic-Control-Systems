import cv2
import time
import ollama

MODEL = "moondream"
NUM_TRIALS = 5
CAM_INDEX = 0

latencies = []

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)

for _ in range(10):
    cap.read()

def run_trial(i):
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Trial {i}: BAD FRAME")
        return None

    # Keep some resizing for speed
    frame = cv2.resize(frame, (640, 480))

    image_path = f"~/Desktop/frame_{i}.jpg"

    cv2.imwrite(image_path, frame)

    start = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        keep_alive="30m",
        options={
            "num_predict": 60
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "Briefly describe the main person or object in the image."
                ),
                "images": [image_path]
            }
        ]
    )

    end = time.perf_counter()

    latency = end - start
    latencies.append(latency)

    print(f"\nTrial {i}")
    print("Latency:", round(latency, 3), "seconds")
    print("Output:", response["message"]["content"])

for i in range(1, NUM_TRIALS + 1):
    run_trial(i)
    time.sleep(1)

cap.release()

print("\n====================")
print("FINAL RESULTS")

print("Average latency:", round(sum(latencies)/len(latencies), 3))
print("Min:", round(min(latencies), 3))
print("Max:", round(max(latencies), 3))