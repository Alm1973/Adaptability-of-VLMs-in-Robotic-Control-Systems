import base64
import glob

import cv2
import ollama

f = sorted(glob.glob("exp_*.jpg"))[0]
img = cv2.imread(f)
small = cv2.resize(img, (480, 360))
ok, buf = cv2.imencode(".jpg", small)
raw = buf.tobytes()
b64 = base64.b64encode(raw).decode()

Q = "Describe what is on the desk in this image."

print("=== A) chat, bytes image, num_gpu=99 ===")
try:
    r = ollama.chat(model="moondream",
                    messages=[{"role": "user", "content": Q, "images": [raw]}],
                    options={"num_gpu": 99, "num_predict": 60})
    print("content:", repr(r["message"]["content"]))
    print("done_reason:", r.get("done_reason"), "eval_count:", r.get("eval_count"))
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== B) chat, base64 str image, default opts ===")
try:
    r = ollama.chat(model="moondream",
                    messages=[{"role": "user", "content": Q, "images": [b64]}])
    print("content:", repr(r["message"]["content"]))
    print("done_reason:", r.get("done_reason"), "eval_count:", r.get("eval_count"))
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== C) generate API, bytes image ===")
try:
    r = ollama.generate(model="moondream", prompt=Q, images=[raw],
                        options={"num_gpu": 99, "num_predict": 60})
    print("response:", repr(r["response"]))
    print("done_reason:", r.get("done_reason"), "eval_count:", r.get("eval_count"))
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== D) text-only chat (does moondream respond at all?) ===")
try:
    r = ollama.chat(model="moondream",
                    messages=[{"role": "user", "content": "Say hello in one word."}])
    print("content:", repr(r["message"]["content"]))
except Exception as e:
    print("ERR", type(e).__name__, e)
