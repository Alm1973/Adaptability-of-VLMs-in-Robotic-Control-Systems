
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

PROMPT = (
    "Look at this image. For EACH object in the list below, say whether it "
    "is visible in the image. Answer with one line per object in the EXACT "
    "format `<object>: yes` or `<object>: no`. Do not add anything else.\n"
    "Objects:\n"
    "water bottle\n"
    "keyboard\n"
    "computer mouse\n"
    "laptop\n"
    "banana"
)


def parse_batched(ans, vocab):
    out = {obj: None for obj in vocab}
    if not ans:
        return out
    low = ans.strip().lower()
    lines = [ln.strip() for ln in low.splitlines() if ln.strip()]

    def verdict_from(segment):
        neg = any(w in segment for w in (" no", ":no", "no,", "no.", "not ",
                                         "absent", "false", "n/a", "none"))
        if segment.strip() in ("no", "-no"):
            neg = True
        pos = any(w in segment for w in ("yes", "present", "visible", "true"))
        if neg and not pos:
            return 0
        if pos and not neg:
            return 1
        if pos and neg:
            return 1 if segment.find("yes") != -1 and (
                segment.find("yes") < segment.find("no")) else 0
        return None

    for obj in sorted(vocab, key=len, reverse=True):
        for ln in lines:
            if obj in ln:
                tail = ln.split(obj, 1)[1]
                v = verdict_from(tail)
                if v is None:
                    v = verdict_from(ln)
                out[obj] = v
                break
    return out


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    results = []
    print(f"{len(frames)} frames x 1 batched call = {len(frames)} VLM queries "
          f"(vs {len(frames) * len(VOCAB)} per-object in Cycle 1)")

    for f in frames:
        idx = idx_of[f]
        img = cv2.imread(f)
        t0 = time.time()
        ans = llm_backend.vision_chat(PROMPT, img, max_tokens=80)
        dt = round(time.time() - t0, 2)
        preds = parse_batched(ans, VOCAB)
        placement = "gpu" if dt < 30 else "cpu_fallback"
        print(f"[{idx:02d}] {dt:5.1f}s  raw={ans!r}")
        for obj in VOCAB:
            gt = GT[idx][obj]
            pred = preds[obj]
            rec = {"idx": idx, "file": f, "obj": obj,
                   "gt": gt, "raw": ans, "pred": pred,
                   "latency_s": dt, "placement": placement,
                   "call_latency_s": dt}
            results.append(rec)
            mark = "OK " if pred == gt else ("?? " if pred is None else "XX ")
            print(f"      {obj:<14} gt={gt} pred={pred} {mark}")
        with open("openvocab_batched_results.json", "w") as fh:
            json.dump(results, fh, indent=2)

    llm_backend.shutdown()
    print("DONE -- wrote openvocab_batched_results.json")


if __name__ == "__main__":
    main()
