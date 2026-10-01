import glob
import cv2
import ollama

frames = sorted(glob.glob("exp_*.jpg"))
f1 = frames[0]

def enc(path):
    img = cv2.imread(path); s = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", s); return buf.tobytes()

img = enc(f1)

PROMPTS = {
    "bare_q_bottle":      "Is there a water bottle in this image?",
    "bare_q_banana":      "Is there a banana in this image?",
    "present_q":          "Does this image contain a keyboard?",
    "how_many":           "How many water bottles are in this image?",
    "caption_objects":    "List every object you can see on the desk.",
    "vqa_style":          "Question: Is there a water bottle in the image? Answer:",
}

for name, p in PROMPTS.items():
    r = ollama.chat(model="moondream",
                    messages=[{"role": "user", "content": p, "images": [img]}],
                    options={"num_gpu": 99, "num_predict": 40})
    print(f"[{name}] eval={r.get('eval_count')} done={r.get('done_reason')}")
    print(f"    -> {r['message']['content'].strip()[:120]!r}\n")
