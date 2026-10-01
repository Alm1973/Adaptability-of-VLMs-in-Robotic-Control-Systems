import cv2
import ollama


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)



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
Analyze this camera frame carefully and describe the scene as accurately as possible.

Your response should include:
1. All visible objects
2. object positions (left, center, right, foreground, background)
3. Colors and features
4. Human presence or actions if visible
5. Lighting conditions and environment type
6. Any uncertainty or unclear areas in the image

Rules:
- Do not invent objects that are not visible

- Keep the description concise but detailed
"""
,
            "images": ["frame.jpg"]
        }
    ]
)


print("\n=== OUTPUT ===\n")
print(response["message"]["content"])
