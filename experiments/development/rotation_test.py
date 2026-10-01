import base64
import glob
import json
import os
import statistics as st
import subprocess
import time
import urllib.request

import cv2

from config import LLAMA_SERVER_PORT
from llm_backend import CUDA_DIR, LIB_DIR, LLAMA_SERVER_EXE, MODEL_BLOB
from openvocab_baseline import GT, PROMPT, VOCAB, parse

PORT = LLAMA_SERVER_PORT
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"

RESOLUTION = (640, 480)
CALLS_PER_PROCESS = 2


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


def health_ok():
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def launch():
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
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):
        if health_ok():
            return p, time.time() - t0
        if p.poll() is not None:
            return None, time.time() - t0
        time.sleep(0.5)
    return None, time.time() - t0


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


def balanced_accuracy(rows):
    bal, per = [], {}
    all_neg = all_fp = 0
    for obj in VOCAB:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos, nneg = tp + fn, tn + fp
        all_neg += nneg
        all_fp += fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        if sens is not None and spec is not None:
            bal.append((sens + spec) / 2)
            per[obj] = round((sens + spec) / 2, 3)
        else:
            per[obj] = None
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, per)


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    enc = {}
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.resize(imgs[f], RESOLUTION))
        enc[f] = base64.b64encode(buf.tobytes()).decode()

    print(f"rotation test: restart every {CALLS_PER_PROCESS} calls, "
          f"{RESOLUTION[0]}x{RESOLUTION[1]}, {len(frames)*len(VOCAB)} queries\n")

    kill_servers()
    time.sleep(1)
    rows, call_lat, restart_lat = [], [], []
    degen = 0
    since_restart = CALLS_PER_PROCESS
    proc = None
    wall0 = time.time()

    for f in frames:
        idx = idx_of[f]
        for obj in VOCAB:
            if since_restart >= CALLS_PER_PROCESS:
                kill_servers()
                time.sleep(0.5)
                proc, ready_s = launch()
                restart_lat.append(ready_s)
                since_restart = 0
                if proc is None:
                    print("  SERVER FAILED TO START")
                    continue
            try:
                out, dt = call(enc[f], PROMPT.format(obj=obj))
            except Exception as e:
                print(f"  ERROR {e}")
                out, dt = "", 0.0
            since_restart += 1
            call_lat.append(dt)
            d = is_degenerate(out)
            degen += d
            rows.append({"idx": idx, "obj": obj,
                         "gt": GT[idx][obj], "pred": parse(out)})
        got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
        print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")

    wall = time.time() - wall0
    kill_servers()
    macro, fp_rate, per = balanced_accuracy(rows)
    n = len(rows)
    out = {
        "macro_bal_acc": round(macro, 3) if macro else None,
        "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "degenerate": degen, "n_queries": n, "per_object": per,
        "med_call_s": round(st.median(call_lat), 2) if call_lat else None,
        "med_restart_s": round(st.median(restart_lat), 2) if restart_lat else None,
        "restarts": len(restart_lat),
        "wall_total_s": round(wall, 1),
        "amortized_per_query_s": round(wall / n, 2) if n else None,
    }
    json.dump(out, open("rotation_test.json", "w"), indent=2)

    print("\n===== ROTATION RESULT =====")
    print(f"macro balanced acc : {out['macro_bal_acc']}   (Cycle-1 ref 0.889)")
    print(f"FP rate            : {out['fp_rate']}   (Cycle-1 ref 0.086)")
    print(f"degenerate         : {degen}/{n}   (must be 0)")
    print(f"median call        : {out['med_call_s']}s")
    print(f"median restart     : {out['med_restart_s']}s  "
          f"({out['restarts']} restarts)")
    print(f"AMORTIZED/query    : {out['amortized_per_query_s']}s   "
          f"(CPU path ref 21-80s, GPU-lowres ref 2.5s @ acc 0.632)")
    print(f"per-object         : {per}")


if __name__ == "__main__":
    main()
