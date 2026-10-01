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
ERASE = f"http://127.0.0.1:{PORT}/slots/0?action=erase"

RESOLUTION = (640, 480)
IMG_MIN_TOKENS = 1024
N_CALLS = 20


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


def launch(logpath):
    cmd = [
        NEW_EXE, "--model", MODEL, "--mmproj", MMPROJ,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "-c", "4096", "-np", "1",
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


def erase_slot():
    try:
        req = urllib.request.Request(
            ERASE, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except Exception as e:
        return f"ERR {e}"


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


def run(enc_list, use_erase):
    deg, first, lat, trace = 0, None, [], []
    for i in range(N_CALLS):
        if use_erase:
            erase_slot()
        try:
            out, dt = call(enc_list[i % len(enc_list)],
                           PROMPT.format(obj="keyboard"))
        except Exception as e:
            print(f"    call {i+1} ERROR {e}")
            out, dt = "", 0.0
        lat.append(dt)
        fr = vram_free()
        trace.append(fr)
        d = is_degenerate(out)
        if d and first is None:
            first = i + 1
        deg += d
        print(f"    {i+1:2d}/{N_CALLS} {dt:5.1f}s free={fr:5d}MiB "
              f"{'DEGEN' if d else 'ok   '} {out[:26]!r}")
    return deg, first, lat, trace


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    enc_list = [encode(imgs[f], RESOLUTION) for f in frames]
    print(f"erase test @ {RESOLUTION[0]}x{RESOLUTION[1]}, "
          f"img-min-tokens {IMG_MIN_TOKENS}, {N_CALLS} calls/config\n")

    summary = {}
    for name, use_erase in (("control_no_erase", False), ("erase", True)):
        kill()
        print(f"##### {name} #####")
        p = launch(f"srv_erase_{name}.log")
        if p is None:
            print("  SERVER FAILED TO START\n")
            summary[name] = {"error": "no_start"}
            continue
        fa = vram_free()
        print(f"  VRAM free after load: {fa} MiB")
        deg, first, lat, trace = run(enc_list, use_erase)
        summary[name] = {
            "erase": use_erase, "degen": deg, "n": N_CALLS,
            "first_lock_at": first, "vram_after_load": fa,
            "vram_first_call": trace[0], "vram_last_call": trace[-1],
            "vram_min": min(trace),
            "med_lat_s": round(st.median(lat), 2),
        }
        json.dump(summary, open("erase_test.json", "w"), indent=2)
        drift = trace[0] - trace[-1]
        print(f"  -> degen {deg}/{N_CALLS} lock@{first}  "
              f"VRAM {trace[0]} -> {trace[-1]} MiB (drift {drift:+d})\n")

    win = summary.get("erase", {})
    if win.get("degen") == 0:
        print("##### ERASE IS CLEAN -- FULL ACCURACY RUN #####")
        kill()
        p = launch("srv_erase_acc.log")
        rows, alat, deg = [], [], 0
        for f in frames:
            idx = idx_of[f]
            b64 = encode(imgs[f], RESOLUTION)
            for obj in VOCAB:
                erase_slot()
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
        summary["erase"].update({
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "acc_degen": deg, "per_object": per,
            "med_acc_lat_s": round(st.median(alat), 2),
            "total_lat_s": round(sum(alat), 1),
        })
        json.dump(summary, open("erase_test.json", "w"), indent=2)
        s = summary["erase"]
        print(f"\n  *** macro {s['macro_bal_acc']} FP {s['fp_rate']} "
              f"degen {deg} med {s['med_acc_lat_s']}s ***")

    kill()
    print("\n===== ERASE TEST SUMMARY =====")
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:<18} FAILED_TO_START")
            continue
        print(f"{name:<18} degen {s['degen']}/{s['n']} lock@{s['first_lock_at']} "
              f"VRAM {s['vram_first_call']}->{s['vram_last_call']}MiB "
              f"macro {s.get('macro_bal_acc','-')} "
              f"med {s.get('med_acc_lat_s', s['med_lat_s'])}s")
    print("\nRefs: CPU macro 0.889 @21-80s | GPU@480x360 macro 0.632 @2.5s")


if __name__ == "__main__":
    main()
