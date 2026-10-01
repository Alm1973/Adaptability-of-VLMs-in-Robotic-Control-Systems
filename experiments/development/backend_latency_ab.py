import base64
import glob
import json
import os
import statistics as st
import subprocess
import time
import urllib.request

import cv2
import numpy as np

OLLAMA_EXE = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
OLLAMA_LIB = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama"
OLLAMA_CUDA = OLLAMA_LIB + r"\cuda_v13"
OLLAMA_BLOB = r"<HOME>\.ollama\models\blobs\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d"

NEW_DIR = r"<HOME>\llamacpp-b10326\bin12"
NEW_EXE = NEW_DIR + r"\llama-server.exe"
NEW_MODEL = r"<HOME>\llamacpp-b10326\models\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
NEW_MMPROJ = r"<HOME>\llamacpp-b10326\models\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

PROMPT = "Is there a keyboard in this image? Answer yes or no."
WH = (480, 360)
N = 15
MAX_TOKENS = 20
SEED = 1234


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def health_ok(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    subprocess.run(["ollama", "stop", "qwen2.5vl:3b"], capture_output=True)
    time.sleep(2)


def launch(cmd, cwd, path_prepend, logpath, port):
    env = dict(os.environ)
    env["PATH"] = path_prepend + ";" + env.get("PATH", "")
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=log)
    for _ in range(240):
        if health_ok(port):
            return p
        if p.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def launch_ollama():
    cmd = [
        OLLAMA_EXE, "--model", OLLAMA_BLOB,
        "--port", "18080", "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", "2048", "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", OLLAMA_BLOB,
        "--no-mmap", "--flash-attn", "off",
        "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    return launch(cmd, OLLAMA_CUDA, OLLAMA_LIB + ";" + OLLAMA_CUDA,
                  "srv_ab_ollama.log", 18080)


def launch_b10326():
    cmd = [
        NEW_EXE, "--model", NEW_MODEL,
        "--port", "18081", "--host", "127.0.0.1",
        "--no-webui", "-c", "2048", "-np", "1",
        "--mmproj", NEW_MMPROJ,
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512",
    ]
    return launch(cmd, NEW_DIR, NEW_DIR, "srv_ab_b10326.log", 18081)


def build_unique_images():
    rng = np.random.default_rng(SEED)
    frames = sorted(glob.glob("exp_*.jpg"))
    bases = [cv2.imread(f) for f in frames]
    imgs = []
    for i in range(N):
        base = bases[i % len(bases)]
        h, w = base.shape[:2]
        cw, ch = int(w * 0.9), int(h * 0.9)
        x = int(rng.integers(0, w - cw + 1))
        y = int(rng.integers(0, h - ch + 1))
        crop = base[y:y + ch, x:x + cw]
        img = cv2.resize(crop, WH)
        delta = int(rng.integers(-25, 26))
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
        imgs.append(img)
    return imgs


def encode_b64(img):
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


def call(port, b64):
    payload = json.dumps({
        "model": "qwen2.5vl", "max_tokens": MAX_TOKENS, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    dt = time.time() - t0
    return body["choices"][0]["message"]["content"].strip(), dt


def run_backend(name, launcher, port, b64s):
    kill_all()
    print(f"\n===== {name}: launching FRESH server on :{port} =====")
    p = launcher()
    if p is None:
        print(f"  SERVER FAILED TO START -- see srv_ab_{name}.log")
        return None
    print("  healthy; running", N, "unique-image calls")
    lat, degen, texts = [], 0, []
    for i, b64 in enumerate(b64s):
        try:
            out, dt = call(port, b64)
        except Exception as e:
            print(f"  {i+1:2d} ERROR {e}")
            texts.append(f"ERR:{e}")
            continue
        d = is_degenerate(out)
        degen += d
        lat.append(dt)
        texts.append(out)
        print(f"  {i+1:2d}/{N}  {dt:6.2f}s  {'DEGEN' if d else 'ok   '} {out[:30]!r}")
    kill_all()
    if not lat:
        return None
    tail = lat[1:] if len(lat) > 1 else lat
    res = {
        "n": len(lat), "degen": degen,
        "cold_call1_s": round(lat[0], 3),
        "median_all_s": round(st.median(lat), 3),
        "median_tail_s": round(st.median(tail), 3),
        "p90_tail_s": round(sorted(tail)[int(0.9 * len(tail)) - 1], 3),
        "max_s": round(max(lat), 3),
        "min_s": round(min(lat), 3),
    }
    print(f"  -> {json.dumps(res)}")
    return res


def gpu_evidence(logpath):
    try:
        lines = open(logpath, errors="ignore").read().splitlines()
    except Exception:
        return "(no log)"
    hits = [ln.strip() for ln in lines
            if "per token" in ln or "prompt eval time" in ln
            or "layers to GPU" in ln or "CUDA0" in ln]
    return hits[-4:] if hits else "(no eval/offload lines found)"


def main():
    b64s = [encode_b64(im) for im in build_unique_images()]
    print(f"built {len(b64s)} unique {WH[0]}x{WH[1]} images (seed {SEED})")

    a = run_backend("ollama", launch_ollama, 18080, b64s)
    b = run_backend("b10326", launch_b10326, 18081, b64s)

    print("\n\n================= SAME-HARNESS LATENCY A/B =================")
    print(f"prompt={PROMPT!r}  res={WH}  N={N}  identical images, fresh servers\n")
    print(f"{'metric':<16} {'A: ollama-bundled':<20} {'B: b10326':<20}")
    if a and b:
        for k in ["n", "degen", "cold_call1_s", "median_all_s",
                  "median_tail_s", "p90_tail_s", "min_s", "max_s"]:
            print(f"{k:<16} {str(a[k]):<20} {str(b[k]):<20}")
        if a["median_tail_s"] and b["median_tail_s"]:
            spd = a["median_tail_s"] / b["median_tail_s"]
            print(f"\nsteady fresh-encode speedup (A median_tail / B median_tail): "
                  f"{spd:.2f}x")
    else:
        print("A or B failed:", "A", bool(a), "B", bool(b))

    print("\nGPU evidence -- A (srv_ab_ollama.log):")
    print("  ", gpu_evidence("srv_ab_ollama.log"))
    print("GPU evidence -- B (srv_ab_b10326.log):")
    print("  ", gpu_evidence("srv_ab_b10326.log"))

    json.dump({"a_ollama": a, "b_b10326": b},
              open("backend_latency_ab.json", "w"), indent=2)
    print("\nwrote backend_latency_ab.json")
    print("\nADOPT RULE: swap llm_backend's server to b10326 ONLY if B's steady "
          "fresh-encode latency is materially below A's (target ~5x) AND B's "
          "degen count is not worse than A's at 480x360.")


if __name__ == "__main__":
    main()
