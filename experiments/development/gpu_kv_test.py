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
ERASE = f"http://127.0.0.1:{PORT}/slots/0?action=erase"
N_CALLS = 25

CONFIGS = {
    "c2048_img1024":      (2048, 1024, False),
    "c4096_img1024":      (4096, 1024, False),
    "c8192_img1024":      (8192, 1024, False),
    "c2048_img256":       (2048,  256, False),
    "c2048_erase":        (2048, 1024, True),
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


def launch(ctx, img_tokens, logpath):
    cmd = [
        LLAMA_SERVER_EXE, "--model", MODEL_BLOB,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", str(ctx), "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", MODEL_BLOB, "--image-min-tokens", str(img_tokens),
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    env = dict(os.environ)
    env["PATH"] = LIB_DIR + ";" + CUDA_DIR + ";" + env.get("PATH", "")
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR, stdout=log, stderr=log)
    for _ in range(150):
        if health_ok():
            return p
        if p.poll() is not None:
            return None
        time.sleep(1)
    return None


def erase_slot():
    try:
        req = urllib.request.Request(ERASE, data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception as e:
        return f"erase_err:{e}"


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
    print(f"{len(enc)} frames, {N_CALLS} calls/config, {len(CONFIGS)} configs")
    results = {}
    for name, (ctx, img_tokens, erase) in CONFIGS.items():
        print(f"\n##### {name}  ctx={ctx} img_tokens={img_tokens} "
              f"erase={erase} #####")
        kill_servers()
        p = launch(ctx, img_tokens, f"srv_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START (see srv log)")
            results[name] = {"error": "no_start"}
            json.dump(results, open("gpu_kv_test.json", "w"), indent=2)
            continue
        degen = 0
        first_lock = None
        clean_before_lock = 0
        for i in range(N_CALLS):
            if erase:
                erase_slot()
            b64 = enc[i % len(enc)]
            try:
                out, dt = call(b64)
            except Exception as e:
                print(f"  call {i+1} ERROR {e}")
                out, dt = "", 0.0
            d = is_degenerate(out)
            if d and first_lock is None:
                first_lock = i + 1
                clean_before_lock = i
            degen += d
            print(f"  {i+1:2d}/{N_CALLS} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:45]!r}")
        results[name] = {
            "degen": degen, "n": N_CALLS, "first_lock_at": first_lock,
            "clean_before_lock": clean_before_lock if first_lock else N_CALLS,
            "ctx": ctx, "img_tokens": img_tokens, "erase": erase,
        }
        json.dump(results, open("gpu_kv_test.json", "w"), indent=2)
    kill_servers()
    print("\n===== SUMMARY =====")
    print(f"{'config':<18} {'ctx':<6} {'img_tok':<8} {'erase':<6} "
          f"{'clean_before_lock':<18} {'first_lock_at'}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<18} FAILED_TO_START")
            continue
        print(f"{name:<18} {r['ctx']:<6} {r['img_tokens']:<8} "
              f"{str(r['erase']):<6} {r['clean_before_lock']:<18} "
              f"{r['first_lock_at']}")


if __name__ == "__main__":
    main()
