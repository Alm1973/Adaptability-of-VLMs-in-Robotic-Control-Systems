import base64
import glob
import json
import os
import subprocess
import time
import urllib.request

import cv2

from config import LLAMA_SERVER_PORT
from llm_backend import CUDA_DIR, LIB_DIR, LLAMA_SERVER_EXE, MODEL_BLOB
from openvocab_batched import PROMPT

PORT = LLAMA_SERVER_PORT
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
N_CALLS = 25

ENVS = {
    "control_no_env":      {},
    "disable_graphs":      {"GGML_CUDA_DISABLE_GRAPHS": "1"},
    "force_cublas":        {"GGML_CUDA_FORCE_CUBLAS": "1"},
    "graphs_off_cublas":   {"GGML_CUDA_DISABLE_GRAPHS": "1",
                            "GGML_CUDA_FORCE_CUBLAS": "1"},
}


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def kill_servers():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    time.sleep(2)


def health_ok():
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def launch(extra_env, logpath):
    cmd = [
        LLAMA_SERVER_EXE, "--model", MODEL_BLOB,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", "2048", "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", MODEL_BLOB, "--image-min-tokens", "1024",
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    env = dict(os.environ)
    env["PATH"] = LIB_DIR + ";" + CUDA_DIR + ";" + env.get("PATH", "")
    env.update({k: str(v) for k, v in extra_env.items()})
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR, stdout=log, stderr=log)
    for _ in range(120):
        if health_ok():
            return p
        if p.poll() is not None:
            return None
        time.sleep(1)
    return None


def encode(path):
    img = cv2.imread(path)
    small = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return base64.b64encode(buf.tobytes()).decode()


def call(b64):
    payload = {
        "model": "qwen2.5vl", "max_tokens": 80, "cache_prompt": False,
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


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    enc = [encode(f) for f in frames]
    print(f"{len(enc)} frames, {N_CALLS} calls/config, {len(ENVS)} env configs")
    results = {}
    for name, extra in ENVS.items():
        print(f"\n##### {name}  env={extra} #####")
        kill_servers()
        p = launch(extra, f"srv_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START (see srv log)")
            results[name] = {"error": "no_start"}
            json.dump(results, open("gpu_cuda_env_test.json", "w"), indent=2)
            continue
        degen = 0
        first_lock = None
        lat = []
        for i in range(N_CALLS):
            b64 = enc[i % len(enc)]
            try:
                out, dt = call(b64)
            except Exception as e:
                print(f"  call {i+1} ERROR {e}")
                out, dt = "", 0.0
            lat.append(dt)
            d = is_degenerate(out)
            if d and first_lock is None:
                first_lock = i + 1
            degen += d
            print(f"  {i+1:2d}/{N_CALLS} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:45]!r}")
        results[name] = {
            "degen": degen, "n": N_CALLS, "first_lock_at": first_lock,
            "med_lat_s": round(sorted(lat)[len(lat) // 2], 2),
        }
        json.dump(results, open("gpu_cuda_env_test.json", "w"), indent=2)
    kill_servers()
    print("\n===== SUMMARY (0 degen across all N = permanent GPU fix) =====")
    for name, r in results.items():
        print(f"{name:<20} {r}")


if __name__ == "__main__":
    main()
