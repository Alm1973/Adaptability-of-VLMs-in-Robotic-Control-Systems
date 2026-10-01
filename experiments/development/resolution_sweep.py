import base64
import glob
import json
import statistics as st
import time
import urllib.request

import cv2

import llm_backend
from config import LLAMA_SERVER_PORT
from openvocab_baseline import GT, PROMPT, VOCAB, parse

URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions"

RESOLUTIONS = [
    (480, 360),
    (640, 480),
    (768, 576),
    (960, 720),
]


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode_at(img, wh):
    small = cv2.resize(img, wh)
    ok, buf = cv2.imencode(".jpg", small)
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
    bal_accs = []
    all_neg = all_fp = 0
    per = {}
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
            bal_accs.append((sens + spec) / 2)
            per[obj] = round((sens + spec) / 2, 3)
        else:
            per[obj] = None
    return (st.mean(bal_accs) if bal_accs else None,
            all_fp / all_neg if all_neg else None, per)


def main():
    if not llm_backend.ensure_running():
        print("server not healthy; aborting")
        return
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}

    summary = {}
    for wh in RESOLUTIONS:
        label = f"{wh[0]}x{wh[1]}"
        print(f"\n##### {label} #####")
        rows = []
        degen = 0
        lat = []
        for f in frames:
            idx = idx_of[f]
            b64 = encode_at(imgs[f], wh)
            for obj in VOCAB:
                prompt = PROMPT.format(obj=obj)
                try:
                    out, dt = call(b64, prompt)
                except Exception as e:
                    print(f"  ERROR {e}")
                    out, dt = "", 0.0
                lat.append(dt)
                if is_degenerate(out):
                    degen += 1
                pred = parse(out)
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": pred})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")
        macro, fp_rate, per = balanced_accuracy(rows)
        summary[label] = {
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "degenerate": degen,
            "per_object": per,
            "med_lat_s": round(st.median(lat), 2),
            "total_lat_s": round(sum(lat), 1),
        }
        print(f"  -> macro bal acc {summary[label]['macro_bal_acc']} "
              f"FP {summary[label]['fp_rate']} degen {degen} "
              f"med {summary[label]['med_lat_s']}s")
        json.dump(summary, open("resolution_sweep.json", "w"), indent=2)

    llm_backend.shutdown()
    print("\n===== SUMMARY (target: macro >= 0.889 @ Cycle-1, 0 degenerate) =====")
    print(f"{'res':<12} {'macro_bal':<11} {'fp_rate':<9} {'degen':<7} "
          f"{'med_s':<8} {'total_s'}")
    for label, s in summary.items():
        print(f"{label:<12} {str(s['macro_bal_acc']):<11} "
              f"{str(s['fp_rate']):<9} {s['degenerate']:<7} "
              f"{s['med_lat_s']:<8} {s['total_lat_s']}")
    print("\nCycle-1 reference: macro 0.889, FP 0.086, total 791.3s (CPU-heavy)")


if __name__ == "__main__":
    main()
