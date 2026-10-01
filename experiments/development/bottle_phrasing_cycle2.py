
import glob
import json
import time

import cv2

import llm_backend

GT_BOTTLE = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1,
}

PHRASINGS = [
    "water bottle",
    "bottle",
    "insulated water bottle",
    "metal water bottle",
    "tumbler",
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


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    results = []
    n = len(frames) * len(PHRASINGS)
    print(f"{len(frames)} frames x {len(PHRASINGS)} phrasings = {n} VLM queries")

    for f in frames:
        idx = idx_of[f]
        img = cv2.imread(f)
        for obj in PHRASINGS:
            t0 = time.time()
            ans = llm_backend.vision_chat(PROMPT.format(obj=obj), img,
                                          max_tokens=10)
            dt = round(time.time() - t0, 2)
            pred = parse(ans)
            placement = "gpu" if dt < 30 else "cpu_fallback"
            gt = GT_BOTTLE[idx]
            rec = {"idx": idx, "file": f, "phrasing": obj,
                   "gt": gt, "raw": ans, "pred": pred,
                   "latency_s": dt, "placement": placement}
            results.append(rec)
            mark = "OK " if pred == gt else "XX "
            if pred is None:
                mark = "?? "
            print(f"  [{idx:02d}] {obj:<22} gt={gt} pred={pred} "
                  f"{mark} {dt:5.1f}s  raw={ans!r}")
            with open("bottle_phrasing_results.json", "w") as fh:
                json.dump(results, fh, indent=2)

    llm_backend.shutdown()
    print("DONE -- wrote bottle_phrasing_results.json")


if __name__ == "__main__":
    main()
