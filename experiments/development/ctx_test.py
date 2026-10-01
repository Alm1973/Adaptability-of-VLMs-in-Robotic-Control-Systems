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
MODEL = MODELS_DIR + r"\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ = MODELS_DIR + r"\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

PORT = 18081
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"

RESOLUTION = (640, 480)
IMG_MIN_TOKENS = 1024
N_PROBE = 12
CTX_VALUES = [2048, 8192, 16384]


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


def launch(ctx, logpath):
    cmd = [
        NEW_EXE, "--model", MODEL, "--mmproj", MMPROJ,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "-c", str(ctx), "-np", "1",
        "--image-min-tokens", str(IMG_MIN_TOKENS),
        "--no-mmap", "--flash-attn", "off",
        "-ngl", "99", "-b", "512", "-ub", "512",
    ]
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
    print(f"ctx test @ {RESOLUTION[0]}x{RESOLUTION[1]}, "
          f"--image-min-tokens {IMG_MIN_TOKENS}, contexts {CTX_VALUES}\n")

    summary = {}
    for ctx in CTX_VALUES:
        name = f"c{ctx}"
        kill()
        print(f"##### {name} #####")
        p = launch(ctx, f"srv_ctx_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START (likely OOM at this context)\n")
            summary[name] = {"error": "no_start"}
            json.dump(summary, open("ctx_test.json", "w"), indent=2)
            continue
        print(f"  loaded. VRAM free: {vram_free()} MiB")
        deg, lat = 0, []
        for i in range(N_PROBE):
            f = frames[i % len(frames)]
            try:
                out, dt = call(encode(imgs[f], RESOLUTION),
                               PROMPT.format(obj="keyboard"))
            except Exception as e:
                print(f"  call {i+1} ERROR {e}")
                out, dt = "", 0.0
            lat.append(dt)
            d = is_degenerate(out)
            deg += d
            print(f"  {i+1:2d}/{N_PROBE} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {out[:32]!r}")
        summary[name] = {"ctx": ctx, "probe_degen": deg, "n_probe": N_PROBE,
                         "med_lat_s": round(st.median(lat), 2) if lat else None}
        json.dump(summary, open("ctx_test.json", "w"), indent=2)
        if deg:
            print(f"  -> {deg}/{N_PROBE} degenerate; not the fix\n")
            continue
        print(f"  -> 0/{N_PROBE} CLEAN! running full accuracy\n")

        rows, alat = [], []
        for f in frames:
            idx = idx_of[f]
            b64 = encode(imgs[f], RESOLUTION)
            for obj in VOCAB:
                try:
                    out, dt = call(b64, PROMPT.format(obj=obj))
                except Exception as e:
                    print(f"  ERROR {e}")
                    out, dt = "", 0.0
                alat.append(dt)
                if is_degenerate(out):
                    deg += 1
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": parse(out)})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")
        macro, fp_rate, per = balanced_accuracy(rows)
        summary[name].update({
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "degen_total": deg, "per_object": per,
            "med_acc_lat_s": round(st.median(alat), 2),
            "total_lat_s": round(sum(alat), 1),
        })
        s = summary[name]
        print(f"\n  *** macro {s['macro_bal_acc']} FP {s['fp_rate']} "
              f"degen {deg} med {s['med_acc_lat_s']}s ***\n")
        json.dump(summary, open("ctx_test.json", "w"), indent=2)
        if deg == 0 and macro and macro >= 0.85:
            print("  *** WINNER: clean + accurate on GPU ***\n")
            break

    kill()
    print("===== CTX TEST SUMMARY =====")
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:<8} FAILED_TO_START")
            continue
        print(f"{name:<8} degen {s['probe_degen']}/{s['n_probe']}  "
              f"macro {s.get('macro_bal_acc')}  "
              f"med {s.get('med_acc_lat_s', s.get('med_lat_s'))}s")
    print("\nRefs: CPU macro 0.889 @21-80s | old-bin GPU@480x360 macro 0.632 @2.5s")


if __name__ == "__main__":
    main()
