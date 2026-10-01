import base64
import glob
import json
import os
import statistics as st
import subprocess
import time
import urllib.request

import cv2

from openvocab_baseline import GT, PROMPT, VOCAB, parse

NEW_DIR = r"<HOME>\llamacpp-b10326\bin12"
NEW_EXE = NEW_DIR + r"\llama-server.exe"
MODELS_DIR = r"<HOME>\llamacpp-b10326\models"
MODEL_BLOB = MODELS_DIR + r"\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ = MODELS_DIR + r"\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
PORT = 18081
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"

RESOLUTIONS = [(640, 480), (480, 360)]
N_PROBE = 12


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
    time.sleep(1)


def launch(logpath):
    cmd = [
        NEW_EXE, "--model", MODEL_BLOB,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "-c", "2048", "-np", "1",
        "--mmproj", MMPROJ,
        "--no-mmap", "--flash-attn", "off", "-ngl", "99",
        "-b", "512", "-ub", "512",
    ]
    env = dict(os.environ)
    env["PATH"] = NEW_DIR + ";" + env.get("PATH", "")
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=NEW_DIR, stdout=log, stderr=log)
    for _ in range(180):
        if health_ok():
            return p
        if p.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def encode(img, wh):
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, wh))
    return base64.b64encode(buf.tobytes()).decode()


def call(b64, prompt, max_tokens=20):
    payload = {
        "model": "qwen2.5vl", "max_tokens": max_tokens, "cache_prompt": False,
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

    kill()
    print(f"launching pinned build: {NEW_EXE}")
    p = launch("srv_newbin.log")
    if p is None:
        print("SERVER FAILED TO START -- srv_newbin.log tail:")
        try:
            print("".join(open("srv_newbin.log").readlines()[-40:]))
        except Exception as e:
            print("(no log)", e)
        return
    print("server healthy\n")

    summary = {}
    for wh in RESOLUTIONS:
        label = f"{wh[0]}x{wh[1]}"
        print(f"##### {label}: degenerate probe ({N_PROBE} calls) #####")
        deg = 0
        for i in range(N_PROBE):
            f = frames[i % len(frames)]
            out, dt = call(encode(imgs[f], wh), PROMPT.format(obj="keyboard"))
            d = is_degenerate(out)
            deg += d
            print(f"  {i+1:2d}/{N_PROBE} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:40]!r}")
        if deg:
            print(f"  -> {deg}/{N_PROBE} degenerate; skipping accuracy run\n")
            summary[label] = {"probe_degen": deg, "macro_bal_acc": None}
            json.dump(summary, open("newbin_test.json", "w"), indent=2)
            continue
        print(f"  -> 0/{N_PROBE} degenerate: CLEAN. running full accuracy\n")

        rows, lat = [], []
        for f in frames:
            idx = idx_of[f]
            b64 = encode(imgs[f], wh)
            for obj in VOCAB:
                out, dt = call(b64, PROMPT.format(obj=obj))
                lat.append(dt)
                if is_degenerate(out):
                    deg += 1
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": parse(out)})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")
        macro, fp_rate, per = balanced_accuracy(rows)
        summary[label] = {
            "probe_degen": 0, "degen_total": deg,
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "per_object": per,
            "med_lat_s": round(st.median(lat), 2),
            "total_lat_s": round(sum(lat), 1),
        }
        s = summary[label]
        print(f"  -> macro {s['macro_bal_acc']} FP {s['fp_rate']} "
              f"degen {deg} med {s['med_lat_s']}s\n")
        json.dump(summary, open("newbin_test.json", "w"), indent=2)

    kill()
    print("===== PINNED b10326 SUMMARY =====")
    print(f"{'res':<12} {'probe_deg':<11} {'macro_bal':<11} {'fp_rate':<9} "
          f"{'med_s':<8} {'total_s'}")
    for label, s in summary.items():
        print(f"{label:<12} {s.get('probe_degen'):<11} "
              f"{str(s.get('macro_bal_acc')):<11} {str(s.get('fp_rate')):<9} "
              f"{str(s.get('med_lat_s')):<8} {s.get('total_lat_s')}")
    print("\nOLD binary refs: @480x360 macro 0.632 clean 2.5s | "
          "@640x480 CORRUPTS | CPU macro 0.889 @21-80s")


if __name__ == "__main__":
    main()
