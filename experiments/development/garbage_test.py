import cv2
import ollama

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)
for _ in range(5):
    cap.read()

prompt = (
    "You are controlling a camera-mounted robotic arm that has lost track "
    "of a red cup. Look at this image and respond with exactly one action "
    "from this list: scan_left, scan_right, tilt_up, tilt_down, object_found. "
    "If the cup is not visible anywhere in the image, pick a direction you "
    "have NOT tried yet based on the scan history below. "
    "Respond with only the action word, nothing else."
)
prompt += "\n\nScan history this session: Already tried: scan_right (1x), scan_left (1x). NOT yet tried: tilt_up, tilt_down."

fails = 0
for i in range(15):
    ret, frame = cap.read()
    small = cv2.resize(frame, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    r = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[{"role": "user", "content": prompt, "images": [buf.tobytes()]}],
        options={"num_ctx": 2048},
    )
    text = r["message"]["content"].strip()
    is_garbage = text.count("@") > 5
    if is_garbage:
        fails += 1
    label = "GARBAGE" if is_garbage else "OK"
    print(f"call {i+1}: {label} -> {text[:40]!r}")

cap.release()
print(f"\nFailure rate: {fails}/15")
