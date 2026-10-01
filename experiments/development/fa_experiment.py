
import base64
import json
import subprocess
import time
import urllib.request

import cv2

LLAMA_SERVER = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
LIB_DIR = r"<HOME>\AppData\Local\Programs\Ollama\lib\ollama"
CUDA_DIR = LIB_DIR + r"\cuda_v13"
BLOB = r"<HOME>\.ollama\models\blobs\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d"
PORT = 18080
CALLS_PER_CONFIG = 4

PROMPT = ("What is the main object visible in this image? Answer with a "
          "short phrase naming the object.")


def make_image_b64():
    frame = cv2.imread("obj2_02_verify_base_plus.jpg")
    small = cv2.resize(frame, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return base64.b64encode(buf.tobytes()).decode()


def wait_ready(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def probe(image_b64):
    payload = json.dumps({
        "model": "qwen2.5vl",
        "max_tokens": 20,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip()


def run_config(fa_mode, image_b64):
    print(f"\n=== --flash-attn {fa_mode} ===")
    cmd = [
        LLAMA_SERVER, "--model", BLOB, "--port", str(PORT),
        "--host", "127.0.0.1", "--no-webui", "--offline",
        "-c", "2048", "-np", "1", "--no-jinja", "--chat-template", "chatml",
        "--mmproj", BLOB, "--image-min-tokens", "1024", "--no-mmap",
        "--flash-attn", fa_mode, "-b", "512", "-ub", "512",
        "--context-shift", "--keep", "4",
    ]
    env = dict(__import__("os").environ)
    env["PATH"] = LIB_DIR + ";" + CUDA_DIR + ";" + env.get("PATH", "")
    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_ready():
            print("  server failed to become ready")
            return []
        results = []
        for i in range(CALLS_PER_CONFIG):
            try:
                out = probe(image_b64)
            except Exception as e:
                out = f"CALL_FAILED: {e}"
            degenerate = out.count("@") >= 5
            results.append((out, degenerate))
            print(f"  call {i+1}/{CALLS_PER_CONFIG}: "
                  f"{'DEGENERATE' if degenerate else 'CLEAN'} -> {out!r}")
        return results
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def main():
    image_b64 = make_image_b64()
    results_off = run_config("off", image_b64)
    results_on = run_config("on", image_b64)

    print("\n=== SUMMARY ===")
    n_deg_off = sum(1 for _, d in results_off if d)
    n_deg_on = sum(1 for _, d in results_on if d)
    print(f"FA off: {n_deg_off}/{len(results_off)} degenerate")
    print(f"FA on:  {n_deg_on}/{len(results_on)} degenerate")


if __name__ == "__main__":
    main()
