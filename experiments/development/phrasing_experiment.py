import glob
import json
import time

import cv2

import llm_backend

BOTTLE_GT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
             7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1}

PHRASINGS = ["water bottle", "insulated water bottle", "bottle"]


def parse_yesno(text):
    if not text:
        return None
    t = text.strip().lower()
    yi, ni = t.find("yes"), t.find("no")
    if yi == -1 and ni == -1:
        return None
    if yi == -1:
        return 0
    if ni == -1:
        return 1
    return 1 if yi < ni else 0


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    imgs = {f: cv2.imread(f) for f in frames}
    results = {}
    for phrasing in PHRASINGS:
        prompt = (f"Is there a {phrasing} in this image? "
                  f"Answer with only one word: yes or no.")
        preds = {}
        print(f"\n=== phrasing: {phrasing!r} ===")
        for f in frames:
            idx = idx_of[f]
            t0 = time.time()
            ans = llm_backend.vision_chat(prompt, imgs[f], max_tokens=10)
            dt = time.time() - t0
            p = parse_yesno(ans)
            preds[idx] = p
            gt = BOTTLE_GT[idx]
            mark = "OK " if p == gt else ("?? " if p is None else "XX ")
            print(f"  [{idx:02d}] {dt:5.1f}s gt={gt} pred={p} {mark} {ans!r}")
        tp = sum(1 for i in preds if BOTTLE_GT[i] == 1 and preds[i] == 1)
        tn = sum(1 for i in preds if BOTTLE_GT[i] == 0 and preds[i] == 0)
        fp = sorted(i for i in preds if BOTTLE_GT[i] == 0 and preds[i] != 0)
        fn = sorted(i for i in preds if BOTTLE_GT[i] == 1 and preds[i] != 1)
        npos = sum(1 for i in BOTTLE_GT.values() if i == 1)
        nneg = sum(1 for i in BOTTLE_GT.values() if i == 0)
        sens = tp / npos
        spec = tn / nneg
        bal = (sens + spec) / 2
        results[phrasing] = {"sens": round(sens, 3), "spec": round(spec, 3),
                             "bal": round(bal, 3), "fp_frames": fp,
                             "fn_frames": fn, "preds": preds}
        print(f"  -> sens {sens:.3f}  spec {spec:.3f}  bal {bal:.3f}  "
              f"FP{fp}  FN{fn}")

    llm_backend.shutdown()
    json.dump(results, open("phrasing_results.json", "w"), indent=2)
    print("\n===== SUMMARY (bottle only; Cycle-1 baseline bal 0.833, "
          "FN[1,2,8]) =====")
    for p, r in results.items():
        print(f"{p:<24} bal {r['bal']:.3f}  sens {r['sens']:.3f}  "
              f"spec {r['spec']:.3f}  FP{r['fp_frames']}  FN{r['fn_frames']}")
    print("wrote phrasing_results.json")


if __name__ == "__main__":
    main()
