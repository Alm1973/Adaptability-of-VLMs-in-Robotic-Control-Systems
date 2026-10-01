import base64
import glob
import json
import os
import random
import statistics as st
import subprocess
import time
import urllib.request

import cv2
import numpy as np

from openvocab_baseline import PROMPT, VOCAB

NEW_DIR = r"<HOME>\llamacpp-b10326\bin12"
NEW_EXE = NEW_DIR + r"\llama-server.exe"
MODELS_DIR = r"<HOME>\llamacpp-b10326\models"
MODEL = MODELS_DIR + r"\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ = MODELS_DIR + r"\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

PORT = 18081
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
RESOLUTION = (480, 360)
N_CALLS = 60

CONFIGS = {
    "control":       [],
    "cache_ram_0":   ["--cache-ram", "0"],
    "no_idle_slots": ["--no-cache-idle-slots"],
    "both":          ["--cache-ram", "0", "--no-cache-idle-slots"],
}


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


def kill():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    time.sleep(2)


def launch(extra, logpath):
    cmd = [
        NEW_EXE, "--model", MODEL, "--mmproj", MMPROJ,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "-c", "2048", "-np", "1",
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512",
    ] + extra
    env = dict(os.environ)
    env["PATH"] = NEW_DIR + ";" + env.get("PATH", "")
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=NEW_DIR, stdout=log, stderr=log)
    for _ in range(240):
        if health_ok():
            return p
        if p.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def make_unique(img, rng, nprng):
    h, w = img.shape[:2]
    dx, dy = rng.randint(0, 40), rng.randint(0, 30)
    cw, ch = w - rng.randint(0, 40), h - rng.randint(0, 30)
    crop = img[dy:dy + ch, dx:dx + cw]
    crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=rng.randint(-18, 18))
    crop[0:4, 0:4] = nprng.randint(0, 255, size=(4, 4, 3)).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, RESOLUTION))
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


def main():
    srcs = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]
    print(f"CACHE-RAM TEST: {N_CALLS} UNIQUE-image calls per config "
          f"@ {RESOLUTION[0]}x{RESOLUTION[1]}")
    print("(unique images = the only condition that has exposed the bug)\n")

    summary = {}
    for name, extra in CONFIGS.items():
        rng, nprng = random.Random(7), np.random.RandomState(7)
        kill()
        print(f"##### {name}  {' '.join(extra) or '(default flags)'} #####")
        p = launch(extra, f"srv_cram_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START\n")
            summary[name] = {"error": "no_start"}
            json.dump(summary, open("cacheram_test.json", "w"), indent=2)
            continue
        deg, first, lat, seen = 0, None, [], set()
        for i in range(N_CALLS):
            b64 = make_unique(srcs[i % len(srcs)], rng, nprng)
            seen.add(hash(b64))
            try:
                out, dt = call(b64, PROMPT.format(obj=VOCAB[i % len(VOCAB)]))
            except Exception as e:
                print(f"  call {i+1} ERROR {e}")
                out, dt = "", 0.0
            lat.append(dt)
            d = is_degenerate(out)
            if d and first is None:
                first = i + 1
            deg += d
            if (i + 1) % 10 == 0 or (d and first == i + 1):
                print(f"  {i+1:3d}/{N_CALLS} {dt:5.2f}s free={vram_free():5d}MiB "
                      f"{'DEGEN' if d else 'ok   '} {out[:18]!r}")
        summary[name] = {
            "flags": extra, "degen": deg, "n": N_CALLS,
            "first_lock_at": first, "unique_images": len(seen),
            "lat_p50": round(sorted(lat)[len(lat) // 2], 2),
            "lat_mean": round(st.mean(lat), 2),
        }
        json.dump(summary, open("cacheram_test.json", "w"), indent=2)
        print(f"  -> degen {deg}/{N_CALLS}  first_lock@{first}  "
              f"unique {len(seen)}\n")
        if deg == 0:
            print(f"  *** {name} SURVIVED {N_CALLS} UNIQUE ENCODES ***\n")

    kill()
    print("===== CACHE-RAM TEST SUMMARY =====")
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:<15} FAILED_TO_START")
            continue
        print(f"{name:<15} degen {s['degen']:>3}/{s['n']}  "
              f"lock@{str(s['first_lock_at']):<5} "
              f"p50 {s['lat_p50']}s")
    print("\nBaseline: default flags corrupt after ~19 unique encodes "
          "at 480x360 (~2 at 640x480).")


if __name__ == "__main__":
    main()
