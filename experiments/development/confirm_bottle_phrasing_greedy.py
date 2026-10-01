
import base64
import glob
import json
import time
import urllib.request

import cv2

import llm_backend
from config import LLAMA_SERVER_PORT

GT_BOTTLE = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1,
}

PHRASINGS = [
    "water bottle",
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


def greedy_chat(prompt, image_bytes, max_tokens=10):
    b64 = base64.b64encode(image_bytes).decode()
    payload = json.dumps({
        "model": "qwen2.5vl",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip()


def greedy_with_recovery(prompt, image_bytes, max_restarts=2, tries_after_restart=2):
    def _one():
        try:
            return greedy_chat(prompt, image_bytes)
        except Exception as e:
            print(f"  [greedy call raised: {e}]")
            return None
    out = None
    for _ in range(2):
        out = _one()
        if out is not None and not llm_backend._is_degenerate(out):
            return out
    for r in range(max_restarts):
        print(f"  [degenerate/err -> restart {r + 1}/{max_restarts} + retry greedy]")
        if not llm_backend.restart():
            continue
        for _ in range(tries_after_restart):
            out = _one()
            if out is not None and not llm_backend._is_degenerate(out):
                return out
    return out if out is not None else ""


def main():
    if not llm_backend.ensure_running():
        print("FATAL: llama-server not healthy")
        return
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    results = []
    print(f"{len(frames)} frames x {len(PHRASINGS)} phrasings (greedy/temp=0)")

    for f in frames:
        idx = idx_of[f]
        img = cv2.imread(f)
        image_bytes = llm_backend._encode_image(img)
        for obj in PHRASINGS:
            t0 = time.time()
            ans = greedy_with_recovery(PROMPT.format(obj=obj), image_bytes)
            dt = round(time.time() - t0, 2)
            pred = parse(ans)
            gt = GT_BOTTLE[idx]
            rec = {"idx": idx, "file": f, "phrasing": obj,
                   "gt": gt, "raw": ans, "pred": pred, "latency_s": dt}
            results.append(rec)
            mark = "OK " if pred == gt else "XX "
            if pred is None:
                mark = "?? "
            print(f"  [{idx:02d}] {obj:<22} gt={gt} pred={pred} "
                  f"{mark} {dt:5.1f}s  raw={ans!r}")
            with open("bottle_phrasing_greedy_results.json", "w") as fh:
                json.dump(results, fh, indent=2)

    llm_backend.shutdown()
    print("DONE -- wrote bottle_phrasing_greedy_results.json")


if __name__ == "__main__":
    main()
