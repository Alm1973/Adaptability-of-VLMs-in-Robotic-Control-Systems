import base64
import glob
import statistics as st
import time

import cv2
import ollama

from openvocab_batched import GT, VOCAB

MODEL = "moondream"


def encode(path):
    img = cv2.imread(path)
    small = cv2.resize(img, (480, 360))
    ok, buf = cv2.imencode(".jpg", small)
    return buf.tobytes()


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def parse_yesno(text):
    t = text.strip().lower()
    yi = t.find("yes")
    ni = t.find("no")
    if yi == -1 and ni == -1:
        return None
    if yi == -1:
        return 0
    if ni == -1:
        return 1
    return 1 if yi < ni else 0


def ask(img, obj):
    t0 = time.time()
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user",
                   "content": f"Is there a {obj} in this image? "
                              f"Answer starting with yes or no.",
                   "images": [img]}],
        options={"num_gpu": 99, "num_predict": 30},
    )
    return (r["message"]["content"].strip(), time.time() - t0,
            r.get("done_reason"), r.get("eval_count"))


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    enc = {f: encode(f) for f in frames}
    print(f"moondream GPU detection: {len(frames)} frames x {len(VOCAB)} objs "
          f"= {len(frames) * len(VOCAB)} calls")

    deg = empty = 0
    lat = []
    rows = []
    for f in frames:
        idx = idx_of[f]
        for obj in VOCAB:
            out, dt, dr, ec = ask(enc[f], obj)
            lat.append(dt)
            if not out:
                empty += 1
                print(f"  [{idx:02d}] {obj:<14} EMPTY done={dr} eval={ec}")
            if is_degenerate(out):
                deg += 1
            pred = parse_yesno(out)
            rows.append({"idx": idx, "obj": obj, "gt": GT[idx][obj], "pred": pred})
        print(f"  frame {idx:02d} done ({st.mean(lat[-5:]):.1f}s/call avg)")

    bal_accs = []
    all_neg = all_fp = 0
    print("\nobject          sens   spec   bal    FP     FN")
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
        bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
        if bal is not None:
            bal_accs.append(bal)
        fmt = lambda x: "n/a " if x is None else f"{x:.2f}"
        print(f"{obj:<15} {fmt(sens)}  {fmt(spec)}  {fmt(bal)}   {fp:<5}  {fn}")
    macro = st.mean(bal_accs) if bal_accs else None
    print(f"\nMACRO balanced accuracy: {macro:.3f}   (qwen baseline 0.889)")
    print(f"Overall FP rate: {all_fp/all_neg:.3f} ({all_fp}/{all_neg})   "
          f"(qwen baseline 0.086)")
    print(f"SUSTAIN: degenerate/empty {deg}/{len(rows)}   empties {empty}   "
          f"lat med {st.median(lat):.1f}s max {max(lat):.1f}s")


if __name__ == "__main__":
    main()
