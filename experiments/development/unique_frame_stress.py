import base64
import glob
import json
import random
import statistics as st
import subprocess
import time
import urllib.request

import cv2
import numpy as np

import llm_backend
from openvocab_baseline import PROMPT, VOCAB

PORT = 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
RESOLUTION = (480, 360)
N_CALLS = 120


def vram_free():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def make_unique(img, rng):
    h, w = img.shape[:2]
    dx, dy = rng.randint(0, 40), rng.randint(0, 30)
    cw, ch = w - rng.randint(0, 40), h - rng.randint(0, 30)
    crop = img[dy:dy + ch, dx:dx + cw]
    beta = rng.randint(-18, 18)
    crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=beta)
    noise = rng.randint(0, 255, size=(4, 4, 3)).astype(np.uint8)
    crop[0:4, 0:4] = noise
    small = cv2.resize(crop, RESOLUTION)
    ok, buf = cv2.imencode(".jpg", small)
    return base64.b64encode(buf.tobytes()).decode()


def call(b64, prompt):
    payload = {
        "model": "qwen2.5vl", "max_tokens": 20, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


def pctl(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    rng = random.Random(1234)
    np_rng = np.random.RandomState(1234)

    class R:
        randint = staticmethod(lambda a, b=None: (
            rng.randint(a, b) if b is not None else np_rng.randint(a)))
    R.randint = lambda a, b=None, size=None: (
        np_rng.randint(a, b, size=size) if size is not None
        else rng.randint(a, b))

    srcs = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]
    if not srcs:
        print("no exp_*.jpg found")
        return
    if not llm_backend.ensure_running():
        print("server not healthy; aborting")
        return

    base = vram_free()
    print(f"UNIQUE-FRAME STRESS: {N_CALLS} calls, every image unique "
          f"(no embedding cache possible)")
    print(f"VRAM free after load: {base} MiB")
    print("expect latency near the ~2.7s fresh-encode figure, "
          "NOT cache-speed ~0.5s\n")

    seen = set()
    lat, trace, degen_at = [], [], []
    for i in range(N_CALLS):
        img = srcs[i % len(srcs)]
        b64 = make_unique(img, R)
        seen.add(hash(b64))
        obj = VOCAB[i % len(VOCAB)]
        try:
            out, dt = call(b64, PROMPT.format(obj=obj))
        except Exception as e:
            print(f"  call {i+1} ERROR {e}")
            out, dt = "", 0.0
        lat.append(dt)
        fr = vram_free()
        trace.append(fr)
        d = is_degenerate(out)
        if d:
            degen_at.append(i + 1)
        if (i + 1) % 10 == 0 or d:
            print(f"  {i+1:3d}/{N_CALLS} {dt:5.2f}s free={fr:5d}MiB "
                  f"{'DEGEN' if d else 'ok   '} {obj:<14} {out[:18]!r}")

    llm_backend.shutdown()
    slow = [x for x in lat if x > 30]
    res = {
        "n_calls": N_CALLS, "unique_images": len(seen),
        "degenerate": len(degen_at), "degenerate_at": degen_at[:20],
        "slow_over_30s": len(slow),
        "vram_first": trace[0], "vram_last": trace[-1],
        "vram_min": min(trace), "vram_drift": trace[0] - trace[-1],
        "lat_p50": round(pctl(lat, .50), 2), "lat_p90": round(pctl(lat, .90), 2),
        "lat_p99": round(pctl(lat, .99), 2), "lat_max": round(max(lat), 2),
        "lat_mean": round(st.mean(lat), 2),
    }
    json.dump(res, open("unique_frame_stress.json", "w"), indent=2)

    print("\n===== UNIQUE-FRAME STRESS RESULT =====")
    print(f"unique images sent : {len(seen)}/{N_CALLS} "
          f"(must equal N -> cache never hit)")
    print(f"degenerate         : {len(degen_at)}   (must be 0)")
    print(f"CPU-scale (>30s)   : {len(slow)}   (must be 0)")
    print(f"VRAM {trace[0]} -> {trace[-1]} MiB, min {min(trace)}, "
          f"drift {res['vram_drift']:+d}")
    print(f"latency p50 {res['lat_p50']}s p90 {res['lat_p90']}s "
          f"p99 {res['lat_p99']}s max {res['lat_max']}s")
    ok = len(degen_at) == 0 and len(slow) == 0 and len(seen) == N_CALLS
    print("VERDICT:", "PASS -- safe for sustained live-video operation"
          if ok else "FAIL -- see counts above")


if __name__ == "__main__":
    main()
