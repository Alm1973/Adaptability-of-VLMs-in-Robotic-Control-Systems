import base64
import glob
import statistics as st
import time

import cv2
import ollama

MODEL = "moondream"


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode(path):
    img = cv2.imread(path)
    small = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return buf.tobytes()


def ask(img, prompt, n_predict=40):
    t0 = time.time()
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [img]}],
        options={"num_gpu": 99, "num_predict": n_predict},
    )
    return r["message"]["content"].strip(), time.time() - t0


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    print(f"moondream GPU sticky-state test over {len(frames)} fresh frames")
    deg = 0
    lat = []
    for i, f in enumerate(frames, 1):
        img = encode(f)
        out, dt = ask(img, "Is there a water bottle in this image? "
                           "Answer yes or no, then name what you see.")
        lat.append(dt)
        d = is_degenerate(out)
        deg += d
        print(f"  [{i:02d}] {dt:5.1f}s {'DEGEN' if d else 'ok   '} {out[:70]!r}")
    print(f"\n=> degenerate {deg}/{len(frames)}   "
          f"lat med {st.median(lat):.1f}s max {max(lat):.1f}s")
    if deg == 0:
        print("VERDICT: moondream survives the full run on GPU (no @@@@). Viable.")
    else:
        print("VERDICT: moondream ALSO degenerates on GPU -- escalate to rollback.")


if __name__ == "__main__":
    main()
