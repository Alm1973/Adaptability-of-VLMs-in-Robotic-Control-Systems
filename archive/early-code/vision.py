import cv2
import ollama


cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera failed")
    exit()


cv2.imwrite("frame.jpg", frame)


response = ollama.chat(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": """
You are controlling a robotic arm with a camera.

Your job:

1. Find the calculator in the image.
2. Estimate its position relative to the CENTER of the image.
3. Output ONLY movement commands to center it.

Allowed actions:
MOVE_UP_10
MOVE_DOWN_10
MOVE_LEFT_10
MOVE_RIGHT_10
STOP

Rules:
- If calculator is top-left then MOVE_UP + MOVE_LEFT
- If top-right then MOVE_UP + MOVE_RIGHT
- If bottom-left then MOVE_DOWN + MOVE_LEFT
- If bottom-right then MOVE_DOWN + MOVE_RIGHT
- If centered then STOP
- Output ONLY commands no explanation
""",
            "images": ["frame.jpg"]
        }
    ]
)

print("\n=== OUTPUT ===\n")
print(response["message"]["content"])