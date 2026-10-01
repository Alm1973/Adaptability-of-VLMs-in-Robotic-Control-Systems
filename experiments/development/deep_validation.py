import base64
import glob
import json
import os
import statistics as st
import subprocess
import time
import urllib.request

import cv2

from llm_backend import CUDA_DIR, LIB_DIR, LLAMA_SERVER_EXE, MODEL_BLOB
from openvocab_baseline import PROMPT, VOCAB

PORT = 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
RESOLUTION = (480, 360)
N_CALLS = 150


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


def health_ok():
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def launch(logpath):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    time.sleep(2)
    cmd = [
        LLAMA_SERVER_EXE, "--model", MODEL_BLOB,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", "2048", "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", MODEL_BLOB,
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    env = dict(os.environ)
    env["PATH"] = LIB_DIR + ";" + CUDA_DIR + ";" + env.get("PATH", "")
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR, stdout=log, stderr=log)
    for _ in range(240):
        if health_ok():
            return p
        if p.poll() is not None:
            return None
        time.sleep(0.5)
    return None


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


def pct(v, q):
    s = sorted(v)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    enc = [(f, base64.b64encode(
        cv2.imencode(".jpg", cv2.resize(cv2.imread(f), RESOLUTION))[1].tobytes()
    ).decode()) for f in frames]
    print(f"DEEP VALIDATION: {N_CALLS} raw calls @ {RESOLUTION[0]}x"
          f"{RESOLUTION[1]}, one process, varied frames x objects\n")

    p = launch("srv_deepval.log")
    if p is None:
        print("SERVER FAILED TO START")
        return
    base_free = vram_free()
    print(f"VRAM free after load: {base_free} MiB\n")

    lat, trace, degens = [], [], []
    first_lock = None
    for i in range(N_CALLS):
        fname, b64 = enc[i % len(enc)]
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
            degens.append(i + 1)
            if first_lock is None:
                first_lock = i + 1
        if (i + 1) % 10 == 0 or d:
            print(f"  {i+1:3d}/{N_CALLS} {dt:5.1f}s free={fr:5d}MiB "
                  f"{'DEGEN' if d else 'ok   '} {obj:<14} {out[:20]!r}")

    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)

    slow = [x for x in lat if x > 30]
    drift = trace[0] - trace[-1]
    res = {
        "n_calls": N_CALLS, "resolution": list(RESOLUTION),
        "degenerate": len(degens), "degenerate_at": degens[:20],
        "first_lock_at": first_lock,
        "slow_over_30s": len(slow),
        "vram_after_load": base_free,
        "vram_first": trace[0], "vram_last": trace[-1],
        "vram_min": min(trace), "vram_drift": drift,
        "lat_mean": round(st.mean(lat), 2),
        "lat_p50": round(pct(lat, 0.50), 2),
        "lat_p90": round(pct(lat, 0.90), 2),
        "lat_p99": round(pct(lat, 0.99), 2),
        "lat_max": round(max(lat), 2),
    }
    json.dump(res, open("deep_validation.json", "w"), indent=2)

    print("\n===== DEEP VALIDATION RESULT =====")
    print(f"calls              : {N_CALLS} (old bug fired at call 3)")
    print(f"degenerate         : {len(degens)}   (must be 0)")
    print(f"CPU-scale (>30s)   : {len(slow)}   (must be 0)")
    print(f"VRAM {trace[0]} -> {trace[-1]} MiB, min {min(trace)}, "
          f"drift {drift:+d}  (flat = not merely delayed)")
    print(f"latency p50 {res['lat_p50']}s  p90 {res['lat_p90']}s  "
          f"p99 {res['lat_p99']}s  max {res['lat_max']}s")
    ok = len(degens) == 0 and len(slow) == 0
    print("VERDICT:", "PASS -- stable under sustained varied load"
          if ok else "FAIL -- see degenerate_at / slow counts")


if __name__ == "__main__":
    main()
