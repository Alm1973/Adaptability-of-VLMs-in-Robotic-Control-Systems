
import glob
import json
import time

import cv2

import llm_backend

VOCAB = ["water bottle", "keyboard", "computer mouse", "laptop", "banana"]
GT = {
    1:  {"water bottle": 1, "keyboard": 1, "computer mouse": 1, "laptop": 0, "banana": 0},
    2:  {"water bottle": 1, "keyboard": 1, "computer mouse": 0, "laptop": 0, "banana": 0},
    3:  {"water bottle": 1, "keyboard": 1, "computer mouse": 0, "laptop": 0, "banana": 0},
    4:  {"water bottle": 1, "keyboard": 1, "computer mouse": 0, "laptop": 0, "banana": 0},
    5:  {"water bottle": 1, "keyboard": 1, "computer mouse": 1, "laptop": 0, "banana": 0},
    6:  {"water bottle": 1, "keyboard": 0, "computer mouse": 0, "laptop": 0, "banana": 0},
    7:  {"water bottle": 0, "keyboard": 1, "computer mouse": 0, "laptop": 1, "banana": 0},
    8:  {"water bottle": 1, "keyboard": 0, "computer mouse": 0, "laptop": 0, "banana": 0},
    9:  {"water bottle": 0, "keyboard": 1, "computer mouse": 1, "laptop": 1, "banana": 0},
    10: {"water bottle": 0, "keyboard": 1, "computer mouse": 0, "laptop": 1, "banana": 0},
    11: {"water bottle": 1, "keyboard": 0, "computer mouse": 0, "laptop": 0, "banana": 0},
    12: {"water bottle": 1, "keyboard": 1, "computer mouse": 1, "laptop": 0, "banana": 0},
}

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
    print(f"{len(frames)} frames x {len(VOCAB)} objects = "
          f"{len(frames) * len(VOCAB)} VLM queries")

    for f in frames:
        idx = idx_of[f]
        img = cv2.imread(f)
        for obj in VOCAB:
            t0 = time.time()
            ans = llm_backend.vision_chat(PROMPT.format(obj=obj), img,
                                          max_tokens=10)
            dt = round(time.time() - t0, 2)
            pred = parse(ans)
            placement = "gpu" if dt < 30 else "cpu_fallback"
            rec = {"idx": idx, "file": f, "obj": obj,
                   "gt": GT[idx][obj], "raw": ans, "pred": pred,
                   "latency_s": dt, "placement": placement}
            results.append(rec)
            gt = GT[idx][obj]
            mark = "OK " if pred == gt else "XX "
            if pred is None:
                mark = "?? "
            print(f"  [{idx:02d}] {obj:<14} gt={gt} pred={pred} "
                  f"{mark} {dt:5.1f}s  raw={ans!r}")
            with open("openvocab_results.json", "w") as fh:
                json.dump(results, fh, indent=2)

    llm_backend.shutdown()
    print("DONE -- wrote openvocab_results.json")


if __name__ == "__main__":
    main()
