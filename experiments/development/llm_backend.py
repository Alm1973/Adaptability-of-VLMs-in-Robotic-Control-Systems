
import base64
import json
import os
import subprocess
import time
import urllib.request

import cv2

from config import LLAMA_SERVER_PORT, GPU_DEGENERATE_RETRIES

LLAMA_SERVER_EXE = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
LIB_DIR = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama"
CUDA_DIR = LIB_DIR + r"\cuda_v13"
MODEL_BLOB = r"<HOME>\.ollama\models\blobs\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d"

OLLAMA_FALLBACK_MODEL = "qwen2.5vl:3b"

ENCODE_BUDGET = 12
_encodes_since_start = 0

GPU_RESTART_CYCLES = 1

_proc = None


def _health_ok(timeout_s=2):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def _evict_ollama_gpu_model():
    try:
        subprocess.run(["ollama", "stop", OLLAMA_FALLBACK_MODEL],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def ensure_running(wait_s=120):
    global _proc
    if _health_ok():
        return True

    if _proc is not None and _proc.poll() is None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None

    _evict_ollama_gpu_model()

    cmd = [
        LLAMA_SERVER_EXE, "--model", MODEL_BLOB,
        "--port", str(LLAMA_SERVER_PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline", "-c", "2048", "-np", "1",
        "--no-jinja", "--chat-template", "chatml",
        "--mmproj", MODEL_BLOB,
        "--no-mmap", "--flash-attn", "off",
        "-ngl", "99",
        "-b", "512", "-ub", "512", "--context-shift", "--keep", "4",
    ]
    env = dict(os.environ)
    env["PATH"] = LIB_DIR + ";" + CUDA_DIR + ";" + env.get("PATH", "")
    try:
        _proc = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[BACKEND] Failed to launch llama-server: {e}")
        return False

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _health_ok():
            print(f"[BACKEND] llama-server ready on port {LLAMA_SERVER_PORT} "
                  f"(GPU/CUDA, flash-attn off)")
            return True
        time.sleep(1)
    print("[BACKEND] llama-server did not become healthy in time")
    return False


def _is_degenerate(text):
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 5:
        return False
    most_common = max(set(stripped), key=stripped.count)
    return stripped.count(most_common) / len(stripped) > 0.8


def restart():
    global _proc, _encodes_since_start
    _encodes_since_start = 0
    print("[BACKEND] Restarting llama-server...")
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None
    time.sleep(2)
    return ensure_running()


def shutdown():
    global _proc
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None


def _encode_image(image_bgr):
    small = cv2.resize(image_bgr, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    if not ok:
        return None
    return buf.tobytes()


def _llama_server_chat(prompt, image_bytes, max_tokens, history=None):
    b64 = base64.b64encode(image_bytes).decode()
    messages = []
    for user_text, assistant_text in (history or []):
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    })
    payload = json.dumps({
        "model": "qwen2.5vl",
        "max_tokens": max_tokens,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip()


def _ollama_cpu_chat(prompt, image_bytes, max_tokens, history=None):
    import ollama
    messages = []
    for user_text, assistant_text in (history or []):
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt,
                     "images": [image_bytes]})
    r = ollama.chat(
        model=OLLAMA_FALLBACK_MODEL,
        messages=messages,
        keep_alive="30m",
        options={"num_ctx": 4096, "num_predict": max_tokens, "num_gpu": 0},
    )
    return r["message"]["content"].strip()


def vision_chat(prompt, image_bgr, max_tokens=20, history=None):
    global _encodes_since_start
    image_bytes = _encode_image(image_bgr)
    if image_bytes is None:
        print("[BACKEND] Image encode failed")
        return None

    if _encodes_since_start >= ENCODE_BUDGET:
        print(f"[BACKEND] encode budget reached "
              f"({_encodes_since_start}/{ENCODE_BUDGET}) -> proactive restart "
              f"(hedge; the degenerate ladder is the real guard)")
        restart()
        _encodes_since_start = 0
    _encodes_since_start += 1

    def _gpu_once():
        try:
            t0 = time.time()
            out = _llama_server_chat(prompt, image_bytes, max_tokens, history)
            dt = time.time() - t0
        except Exception as e:
            print(f"[BACKEND] GPU call raised ({e})")
            return None
        if _is_degenerate(out):
            print(f"[BACKEND] GPU degenerate ({dt:.1f}s)")
            return None
        print(f"[BACKEND] GPU ok ({dt:.1f}s)")
        return out

    if ensure_running():
        for _ in range(2):
            out = _gpu_once()
            if out is not None:
                return out
        for attempt in range(1, GPU_RESTART_CYCLES + 1):
            print(f"[BACKEND] degenerate wedged -> restart {attempt}/"
                  f"{GPU_RESTART_CYCLES} to clear sticky CUDA state")
            if not restart():
                break
            for _ in range(2):
                out = _gpu_once()
                if out is not None:
                    return out

    print("[BACKEND] Falling back to ollama-CPU (slow but clean)")
    try:
        t0 = time.time()
        out = _ollama_cpu_chat(prompt, image_bytes, max_tokens, history)
        print(f"[BACKEND] ollama-CPU call ok ({time.time()-t0:.1f}s)")
        return out
    except Exception as e:
        print(f"[BACKEND] ollama-CPU fallback also failed: {e}")
        return None
