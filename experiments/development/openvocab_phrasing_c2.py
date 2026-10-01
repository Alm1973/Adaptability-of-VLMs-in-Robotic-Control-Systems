
import glob
import json
import time

import cv2

import llm_backend

BOTTLE_GT = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1,
}

PHRASINGS = [
    "water bottle",
    "insulated water bottle",
    "metal water bottle",
]

PROMPT = ("Is there a {obj} in this image? "
          "Answer with only one word: yes or no.")


def parse(ans):
    if ans is None:
        return None
    low = ans.strip().lower()
    if low.startswith("yes"):
        return 1
    if low.startswith("no"):
        return 0
    if "yes" in low and "no" not in low:
        return 1
    if "no" in low and "yes" not in low:
        return 0
    return None


def pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def score(rows):
    print(f"\n{'phrasing':<24} {'sens':<7} {'spec':<7} {'bal_acc':<8} "
          f"{'FP frames':<14} {'FN frames'}")
    for ph in PHRASINGS:
        rs = [r for r in rows if r["phrasing"] == ph]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos, nneg = tp + fn, tn + fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
        fp_fr = sorted(r["idx"] for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn_fr = sorted(r["idx"] for r in rs if r["gt"] == 1 and r["pred"] != 1)
        print(f"{ph:<24} {pct(sens):<7} {pct(spec):<7} {pct(bal):<8} "
              f"{str(fp_fr):<14} {fn_fr}")

    lat = [r["latency_s"] for r in rows]
    none_n = sum(1 for r in rows if r["pred"] is None)
    print(f"\nlatency (s): mean={sum(lat)/len(lat):.1f} "
          f"min={min(lat):.1f} max={max(lat):.1f}  "
          f"new-image-encode(>30s)={sum(1 for x in lat if x > 30)}  "
          f"unparseable={none_n}")


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    results = []
    print(f"{len(frames)} frames x {len(PHRASINGS)} phrasings = "
          f"{len(frames) * len(PHRASINGS)} VLM queries "
          f"(frame-outer for embedding cache reuse)")

    for f in frames:
        idx = idx_of[f]
        img = cv2.imread(f)
        gt = BOTTLE_GT[idx]
        for ph in PHRASINGS:
            t0 = time.time()
            ans = llm_backend.vision_chat(PROMPT.format(obj=ph), img,
                                          max_tokens=10)
            dt = round(time.time() - t0, 2)
            pred = parse(ans)
            rec = {"idx": idx, "file": f, "phrasing": ph,
                   "gt": gt, "raw": ans, "pred": pred, "latency_s": dt}
            results.append(rec)
            mark = "OK " if pred == gt else ("?? " if pred is None else "XX ")
            print(f"  [{idx:02d}] {ph:<24} gt={gt} pred={pred} {mark} "
                  f"{dt:5.1f}s  raw={ans!r}")
            with open("openvocab_phrasing_c2.json", "w") as fh:
                json.dump(results, fh, indent=2)

    llm_backend.shutdown()
    score(results)
    print("\nDONE -- wrote openvocab_phrasing_c2.json")


if __name__ == "__main__":
    main()
