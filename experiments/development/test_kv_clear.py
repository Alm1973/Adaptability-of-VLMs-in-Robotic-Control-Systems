import base64
import glob
import json
import statistics as st
import time
import urllib.request

import cv2

import llm_backend
from config import LLAMA_SERVER_PORT
from openvocab_batched import GT, PROMPT, VOCAB, parse_batched

BASE = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
URL = BASE + "/v1/chat/completions"


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode_b64(path):
    img = cv2.imread(path)
    small = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return base64.b64encode(buf.tobytes()).decode()


def erase_slot(sid=0):
    req = urllib.request.Request(
        BASE + f"/slots/{sid}?action=erase", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except Exception as e:
        return f"ERR {e}"


def call(b64, max_tokens=80):
    payload = {
        "model": "qwen2.5vl", "max_tokens": max_tokens, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


def run(label, frames, enc, idx_of, erase):
    deg = correct = total = 0
    lat = []
    print(f"\n=== {label} (erase_before_each={erase}) ===")
    for f in frames:
        idx = idx_of[f]
        es = erase_slot(0) if erase else "-"
        out, dt = call(enc[f])
        lat.append(dt)
        d = is_degenerate(out)
        deg += d
        preds = parse_batched(out, VOCAB)
        for obj in VOCAB:
            total += 1
            if preds[obj] == GT[idx][obj]:
                correct += 1
        print(f"  [{idx:02d}] erase={es} {dt:5.1f}s "
              f"{'DEGEN' if d else 'ok   '} {out[:50]!r}")
    print(f"  -> degen {deg}/{len(frames)}  acc {correct/total:.3f}  "
          f"lat med {st.median(lat):.1f}s max {max(lat):.1f}s")
    return deg, len(frames)


def main():
    if not llm_backend.ensure_running():
        print("server not healthy")
        return
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    enc = {f: encode_b64(f) for f in frames}
    print("erase endpoint probe:", erase_slot(0))

    run("control: no erase (expect still stuck)", frames, enc, idx_of, erase=False)
    run("FIX: per-call KV erase", frames, enc, idx_of, erase=True)


if __name__ == "__main__":
    main()
