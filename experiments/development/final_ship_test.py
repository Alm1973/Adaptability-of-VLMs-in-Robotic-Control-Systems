import base64
import glob
import json
import os
import statistics as st
import subprocess
import time
import urllib.request

import cv2

from llm_backend import CUDA_DIR, LIB_DIR, LLAMA_SERVER_EXE, MODEL_BLOB
from openvocab_baseline import GT, PROMPT, VOCAB, parse

PORT = 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"

CANDIDATES = [(576, 432), (480, 360)]
N_SOAK = 20


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


def launch(logpath):
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
    log = open(logpath, "w")
    p = subprocess.Popen(cmd, env=env, cwd=CUDA_DIR, stdout=log, stderr=log)
    for _ in range(240):
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

    results = {}
    for wh in CANDIDATES:
        label = f"{wh[0]}x{wh[1]}"
        kill()
        print(f"\n##### {label}: RAW soak {N_SOAK} calls (no retry ladder) #####")
        p = launch(f"srv_ship_{label}.log")
        if p is None:
            print("  SERVER FAILED TO START")
            results[label] = {"error": "no_start"}
            continue
        enc_list = [encode(imgs[f], wh) for f in frames]
        deg, first, lat = 0, None, []
        for i in range(N_SOAK):
            try:
                out, dt = call(enc_list[i % len(enc_list)],
                               PROMPT.format(obj="keyboard"))
            except Exception as e:
                print(f"  call {i+1} ERROR {e}")
                out, dt = "", 0.0
            lat.append(dt)
            d = is_degenerate(out)
            if d and first is None:
                first = i + 1
            deg += d
            print(f"  {i+1:2d}/{N_SOAK} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:26]!r}")
        results[label] = {"soak_degen": deg, "n": N_SOAK,
                          "first_lock_at": first,
                          "med_lat_s": round(st.median(lat), 2)}
        json.dump(results, open("final_ship_test.json", "w"), indent=2)
        if deg:
            print(f"  -> {deg}/{N_SOAK} degenerate (lock@{first}): UNSTABLE\n")
            continue
        print(f"  -> {N_SOAK}/{N_SOAK} CLEAN on raw GPU. accuracy pass\n")

        rows, alat, adeg = [], [], 0
        for f in frames:
            idx = idx_of[f]
            b64 = encode(imgs[f], wh)
            for obj in VOCAB:
                try:
                    out, dt = call(b64, PROMPT.format(obj=obj))
                except Exception as e:
                    print(f"  ERROR {e}")
                    out, dt = "", 0.0
                alat.append(dt)
                if is_degenerate(out):
                    adeg += 1
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": parse(out)})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")
        macro, fp_rate, per = balanced_accuracy(rows)
        results[label].update({
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "acc_degen": adeg, "per_object": per,
            "med_acc_lat_s": round(st.median(alat), 2),
            "total_lat_s": round(sum(alat), 1),
        })
        json.dump(results, open("final_ship_test.json", "w"), indent=2)
        r = results[label]
        print(f"\n  *** {label}: macro {r['macro_bal_acc']} FP {r['fp_rate']} "
              f"degen {adeg}/60 med {r['med_acc_lat_s']}s ***")

    kill()
    print("\n===== FINAL SHIP SUMMARY =====")
    for lb, r in results.items():
        if "error" in r:
            print(f"{lb:<10} FAILED_TO_START")
            continue
        print(f"{lb:<10} soak {r['soak_degen']}/{r['n']} degen  "
              f"macro {r.get('macro_bal_acc','-')}  "
              f"FP {r.get('fp_rate','-')}  "
              f"med {r.get('med_acc_lat_s', r['med_lat_s'])}s")
    print("\nRefs: Cycle-1 CPU macro 0.889 @21-80s | GPU@480x360 macro 0.632")


if __name__ == "__main__":
    main()
