import glob
import statistics as st
import time

import cv2
import ollama

from openvocab_baseline import GT, PROMPT, VOCAB, parse

RESOLUTION = (640, 480)
MODELS = ["qwen2.5vl:3b", "moondream:latest"]


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode(img, wh):
    small = cv2.resize(img, wh)
    ok, buf = cv2.imencode(".jpg", small)
    return buf.tobytes()


def ask(model, prompt, image_bytes):
    t0 = time.time()
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt,
                   "images": [image_bytes]}],
        keep_alive="10m",
        options={"num_ctx": 2048, "num_predict": 20},
    )
    return r["message"]["content"].strip(), time.time() - t0


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
    print(f"{len(frames)} frames x {len(VOCAB)} objects @ "
          f"{RESOLUTION[0]}x{RESOLUTION[1]}, GPU via ollama\n")

    summary = {}
    for model in MODELS:
        print(f"##### {model} #####")
        rows, lat, degen = [], [], 0
        for f in frames:
            idx = idx_of[f]
            b = encode(imgs[f], RESOLUTION)
            for obj in VOCAB:
                try:
                    out, dt = ask(model, PROMPT.format(obj=obj), b)
                except Exception as e:
                    print(f"  ERROR {e}")
                    out, dt = "", 0.0
                lat.append(dt)
                if is_degenerate(out):
                    degen += 1
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": parse(out)})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")
        macro, fp_rate, per = balanced_accuracy(rows)
        summary[model] = {
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "degenerate": degen, "per_object": per,
            "med_lat_s": round(st.median(lat), 2),
            "total_lat_s": round(sum(lat), 1),
        }
        s = summary[model]
        print(f"  -> macro {s['macro_bal_acc']} FP {s['fp_rate']} "
              f"degen {degen}/{len(rows)} med {s['med_lat_s']}s\n")

    print("===== SUMMARY =====")
    print(f"{'model':<18} {'macro_bal':<11} {'fp_rate':<9} {'degen':<8} "
          f"{'med_s':<8} {'total_s'}")
    for m, s in summary.items():
        print(f"{m:<18} {str(s['macro_bal_acc']):<11} {str(s['fp_rate']):<9} "
              f"{s['degenerate']:<8} {s['med_lat_s']:<8} {s['total_lat_s']}")
    print("\nReference: Cycle-1 CPU macro 0.889 / FP 0.086 / 791s total")
    print("Reference: llama-server GPU @480x360 macro 0.632 / FP 0.314 / 2.5s med")
    for m, s in summary.items():
        print(f"per-object {m}: {s['per_object']}")


if __name__ == "__main__":
    main()
