import glob
import statistics as st
import time

import cv2
import ollama

from openvocab_batched import GT, VOCAB

KW = {
    "water bottle": ["bottle", "water bottle", "flask", "tumbler"],
    "keyboard": ["keyboard"],
    "computer mouse": ["mouse"],
    "laptop": ["laptop"],
    "banana": ["banana"],
}
PROMPT = ("Describe this image and list all the objects on the desk, "
          "including any bottle, keyboard, mouse, laptop, or fruit.")


def enc(path):
    img = cv2.imread(path); s = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", s); return buf.tobytes()


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    rows = []
    empties = 0
    lat = []
    for f in frames:
        idx = idx_of[f]
        t0 = time.time()
        r = ollama.chat(model="moondream",
                        messages=[{"role": "user", "content": PROMPT,
                                   "images": [enc(f)]}],
                        options={"num_gpu": 99, "num_predict": 120})
        dt = time.time() - t0
        lat.append(dt)
        cap = r["message"]["content"].strip().lower()
        if not cap:
            empties += 1
        print(f"[{idx:02d}] {dt:4.1f}s eval={r.get('eval_count')} {cap[:90]!r}")
        for obj in VOCAB:
            pred = 1 if any(k in cap for k in KW[obj]) else 0
            rows.append({"idx": idx, "obj": obj, "gt": GT[idx][obj], "pred": pred})

    bal_accs = []
    all_neg = all_fp = 0
    print("\nobject          sens   spec   bal    FP  FN")
    for obj in VOCAB:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 1)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 0)
        npos, nneg = tp + fn, tn + fp
        all_neg += nneg; all_fp += fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
        if bal is not None:
            bal_accs.append(bal)
        fmt = lambda x: "n/a " if x is None else f"{x:.2f}"
        print(f"{obj:<15} {fmt(sens)}  {fmt(spec)}  {fmt(bal)}   {fp}   {fn}")
    print(f"\nMACRO balanced accuracy: {st.mean(bal_accs):.3f}   (qwen 0.889)")
    print(f"Overall FP rate: {all_fp/all_neg:.3f} ({all_fp}/{all_neg})   (qwen 0.086)")
    print(f"SUSTAIN: empties {empties}/12   lat med {st.median(lat):.1f}s "
          f"max {max(lat):.1f}s (ALL 12 GPU, no @@@@)")


if __name__ == "__main__":
    main()
