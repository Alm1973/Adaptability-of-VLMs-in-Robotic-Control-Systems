import base64
import glob
import json
import statistics as st
import time
import urllib.request

import cv2

import llm_backend
from config import LLAMA_SERVER_PORT
from openvocab_batched import GT, PROMPT, VOCAB, parse_batched

URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions"


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode_b64(path):
    img = cv2.imread(path)
    small = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return base64.b64encode(buf.tobytes()).decode()


def call(b64, sampler, max_tokens=80):
    payload = {
        "model": "qwen2.5vl",
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }
    payload.update(sampler)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


DRY = {"dry_multiplier": 0.8, "dry_base": 1.75,
       "dry_allowed_length": 2, "dry_penalty_last_n": -1}

CONFIGS = {
    "baseline_default": {},
    "greedy_t0":        {"temperature": 0.0},
}


def run_config(name, sampler, enc, idx_of):
    deg = 0
    correct = total = 0
    none_n = 0
    lat = []
    print(f"\n=== {name}  {sampler} ===")
    for f in sorted(enc):
        idx = idx_of[f]
        out, dt = call(enc[f], sampler)
        lat.append(dt)
        d = is_degenerate(out)
        deg += d
        preds = parse_batched(out, VOCAB)
        for obj in VOCAB:
            total += 1
            if preds[obj] is None:
                none_n += 1
            if preds[obj] == GT[idx][obj]:
                correct += 1
        flag = "DEGEN" if d else "ok   "
        print(f"  [{idx:02d}] {dt:5.1f}s {flag} {out[:55]!r}")
    return {
        "degen": deg, "n_frames": len(enc),
        "raw_acc": round(correct / total, 3), "unparseable": none_n,
        "med_lat_s": round(st.median(lat), 2),
        "max_lat_s": round(max(lat), 2),
    }


def main():
    if not llm_backend.ensure_running():
        print("server not healthy; aborting")
        return
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    print(f"{len(frames)} fresh-encode frames x {len(CONFIGS)} configs")
    enc = {f: encode_b64(f) for f in frames}

    summary = {}
    for name, samp in CONFIGS.items():
        summary[name] = run_config(name, samp, enc, idx_of)
        json.dump(summary, open("gpu_degen_diag.json", "w"), indent=2)

    print("\n===== SUMMARY (degenerate rate is the target metric) =====")
    print(f"{'config':<20} {'degen':<10} {'raw_acc':<8} {'unparse':<8} "
          f"{'med_s':<7} {'max_s'}")
    for name, s in summary.items():
        print(f"{name:<20} {s['degen']}/{s['n_frames']:<7} {s['raw_acc']:<8} "
              f"{s['unparseable']:<8} {s['med_lat_s']:<7} {s['max_lat_s']}")
    print("\nwrote gpu_degen_diag.json")


if __name__ == "__main__":
    main()
