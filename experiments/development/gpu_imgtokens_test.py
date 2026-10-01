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

CONFIGS = {
    "gpu_img256":  (256,  False, 15),
    "gpu_img512":  (512,  False, 15),
    "gpu_default": (None, False, 15),
    "cpu_img256":  (256,  True,   6),
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


def launch(img_tokens, mmproj_cpu, logpath):
    cmd = [
        LLAMA_SERVER_EXE, "--model", MODEL_BLOB,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", "2048", "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", MODEL_BLOB,
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    if img_tokens is not None:
        cmd += ["--image-min-tokens", str(img_tokens)]
    if mmproj_cpu:
        cmd += ["--no-mmproj-offload"]
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
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    enc = [encode(f) for f in frames]
    print(f"{len(enc)} frames | matrix: {list(CONFIGS)}")
    results = {}
    for name, (img_tokens, mmproj_cpu, n_calls) in CONFIGS.items():
        print(f"\n##### {name}  img_tokens={img_tokens} "
              f"mmproj_cpu={mmproj_cpu} calls={n_calls} #####")
        kill_servers()
        p = launch(img_tokens, mmproj_cpu, f"srv_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START (see srv log)")
            results[name] = {"error": "no_start"}
            json.dump(results, open("gpu_imgtokens_test.json", "w"), indent=2)
            continue
        degen = 0
        first_lock = None
        lat = []
        for i in range(n_calls):
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
            print(f"  {i+1:2d}/{n_calls} {dt:6.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:45]!r}")
        good = [l for l in lat if l > 0]
        results[name] = {
            "degen": degen, "n": n_calls, "first_lock_at": first_lock,
            "clean": degen == 0,
            "med_lat_s": round(sorted(good)[len(good) // 2], 2) if good else None,
            "max_lat_s": round(max(good), 2) if good else None,
            "img_tokens": img_tokens, "mmproj_cpu": mmproj_cpu,
        }
        json.dump(results, open("gpu_imgtokens_test.json", "w"), indent=2)
    kill_servers()

    print("\n===== SUMMARY =====")
    print(f"{'config':<14} {'img_tok':<9} {'vision':<7} {'degen':<8} "
          f"{'lock@':<7} {'med_s':<8} {'VERDICT'}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<14} FAILED_TO_START")
            continue
        vision = "CPU" if r["mmproj_cpu"] else "GPU"
        verdict = "CLEAN" if r["clean"] else "locks"
        print(f"{name:<14} {str(r['img_tokens']):<9} {vision:<7} "
              f"{r['degen']}/{r['n']:<5} {str(r['first_lock_at']):<7} "
              f"{str(r['med_lat_s']):<8} {verdict}")
    winners = [n for n, r in results.items()
               if "error" not in r and r["clean"] and not r["mmproj_cpu"]]
    print(f"\nFULL-GPU clean configs: {winners or 'NONE'}")


if __name__ == "__main__":
    main()
