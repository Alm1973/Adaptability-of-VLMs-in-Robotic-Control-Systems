import glob
import json
import re
import statistics as st
import time

import cv2

import llm_backend
from openvocab_baseline import GT, VOCAB

VARIANTS = {
    "baseline": "Is there a {obj} in this image? Answer with only one word: yes or no.",
    "skeptical": ("Look carefully at this image. Is a {obj} CLEARLY VISIBLE? "
                  "Answer 'yes' only if you can actually see it in the image. "
                  "If you are unsure or it is not there, answer 'no'. "
                  "Reply with one word: yes or no."),
    "neg_primed": ("This image contains only a few objects. Most objects you "
                   "might be asked about are NOT present. Is a {obj} actually "
                   "visible in this image? Reply with one word: yes or no."),
    "evidence": ("Look at this image. First name the main objects you can "
                 "actually see. Then decide whether a {obj} is among them. "
                 "Finish your reply with exactly 'ANSWER: yes' or "
                 "'ANSWER: no'."),
}
MAX_TOKENS = {"baseline": 20, "skeptical": 20, "neg_primed": 20,
              "evidence": 80}


def parse_answer(text, variant):
    if not text:
        return None
    t = text.strip().lower()
    if variant == "evidence":
        m = re.search(r"answer\s*:\s*(yes|no)", t)
        if m:
            return 1 if m.group(1) == "yes" else 0
        t = t[-40:]
    if re.search(r"\b(no|not|absent|none)\b", t) and not re.search(r"\byes\b", t):
        return 0
    if re.search(r"\byes\b", t):
        return 1
    if re.search(r"\bno\b", t):
        return 0
    return None


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
    print(f"PROMPT VARIANTS: {len(VARIANTS)} prompts x {len(frames)} frames "
          f"x {len(VOCAB)} objects\n")

    summary = {}
    for name, tmpl in VARIANTS.items():
        print(f"##### {name} #####")
        rows, lat, unparsed = [], [], 0
        for f in frames:
            idx = idx_of[f]
            for obj in VOCAB:
                t0 = time.time()
                out = llm_backend.vision_chat(tmpl.format(obj=obj), imgs[f],
                                              max_tokens=MAX_TOKENS[name])
                lat.append(time.time() - t0)
                pred = parse_answer(out, name)
                if pred is None:
                    unparsed += 1
                rows.append({"idx": idx, "obj": obj,
                             "gt": GT[idx][obj], "pred": pred})
            got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
            print(f"  [{idx:02d}] {got}/{len(VOCAB)}")
        macro, fp_rate, per = balanced_accuracy(rows)
        summary[name] = {
            "macro_bal_acc": round(macro, 3) if macro else None,
            "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
            "unparseable": unparsed, "per_object": per,
            "med_lat_s": round(st.median(lat), 2),
        }
        json.dump(summary, open("prompt_variants_test.json", "w"), indent=2)
        s = summary[name]
        print(f"  -> macro {s['macro_bal_acc']}  FP {s['fp_rate']}  "
              f"unparsed {unparsed}  med {s['med_lat_s']}s\n")

    llm_backend.shutdown()
    print("===== PROMPT VARIANT SUMMARY =====")
    print(f"{'variant':<12} {'macro_bal':<11} {'fp_rate':<9} "
          f"{'unparsed':<10} {'med_s'}")
    for name, s in summary.items():
        print(f"{name:<12} {str(s['macro_bal_acc']):<11} "
              f"{str(s['fp_rate']):<9} {s['unparseable']:<10} "
              f"{s['med_lat_s']}")
    print("\nBaseline to beat: macro ~0.70, FP 0.343 "
          "(FP weighted heavily for this robot).")
    best = max((s for s in summary.values() if s["macro_bal_acc"]),
               key=lambda s: s["macro_bal_acc"], default=None)
    if best:
        for n, s in summary.items():
            if s is best:
                print(f"BEST macro: {n} ({s['macro_bal_acc']}, FP {s['fp_rate']})")


if __name__ == "__main__":
    main()
