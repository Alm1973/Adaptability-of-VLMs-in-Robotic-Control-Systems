import base64
import glob
import json
import os
import subprocess
import time
import urllib.request

import cv2

DIR = r"<HOME>\avi\llamacpp_b10310"
EXE = DIR + r"\llama-server.exe"
BLOB = (r"<HOME>\.ollama\models\blobs"
        r"\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d")
PORT = 18090
URL = f"http://127.0.0.1:{PORT}"

PROMPT = ("Is there a water bottle in this image? For each of these objects "
          "say yes or no, one per line: water bottle, keyboard, computer "
          "mouse, laptop, banana.")


def health():
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def launch():
    cmd = [EXE, "--model", BLOB, "--mmproj", BLOB, "--port", str(PORT),
           "--host", "127.0.0.1", "--no-webui", "-c", "2048", "-np", "1",
           "-ngl", "99", "-b", "512", "-ub", "512"]
    env = dict(os.environ)
    env["PATH"] = DIR + ";" + env.get("PATH", "")
    logf = open(os.path.join(DIR, "b10310_server.log"), "w")
    p = subprocess.Popen(cmd, env=env, cwd=DIR, stdout=logf,
                         stderr=subprocess.STDOUT)
    for _ in range(120):
        if health():
            print("[server healthy]")
            return p
        if p.poll() is not None:
            print(f"[server EXITED early, code {p.returncode} -- see "
                  f"b10310_server.log]")
            return p
        time.sleep(1)
    print("[server never healthy]")
    return p


def is_degen(t):
    if not t:
        return True
    s = t.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def enc(path):
    img = cv2.imread(path)
    s = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", s)
    return base64.b64encode(buf.tobytes()).decode()


def call(b64):
    payload = {"model": "qwen2.5vl", "max_tokens": 80, "cache_prompt": False,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": PROMPT},
                   {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


def main():
    p = launch()
    if not health():
        print("ABORT: server not up")
        return
    frames = sorted(glob.glob("exp_*.jpg"))
    deg = 0
    for i, f in enumerate(frames, 1):
        try:
            out, dt = call(enc(f))
        except Exception as e:
            print(f"[{i:02d}] CALL FAILED: {e}")
            deg += 1
            continue
        d = is_degen(out)
        deg += d
        print(f"[{i:02d}] {dt:5.1f}s {'DEGEN' if d else 'ok   '} "
              f"{out[:55]!r}")
    print(f"\nb10310 result: degenerate {deg}/{len(frames)}   "
          f"({'FIXED' if deg == 0 else 'still broken'})")
    p.terminate()


if __name__ == "__main__":
    main()
