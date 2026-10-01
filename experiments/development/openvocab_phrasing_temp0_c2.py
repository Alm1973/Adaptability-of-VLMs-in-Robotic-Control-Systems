
import base64
import glob
import json
import time
import urllib.request

import cv2

import llm_backend
from config import LLAMA_SERVER_PORT

BOTTLE_GT = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
    7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1,
}
PHRASINGS = [
    "water bottle",
    "insulated water bottle",
    "metal water bottle",
]
FRAME_SUBSET = {1, 2, 7, 8, 9, 10}
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


def _greedy_gpu_call(prompt, image_bytes, max_tokens=10):
    b64 = base64.b64encode(image_bytes).decode()
    payload = json.dumps({
        "model": "qwen2.5vl",
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
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


def _greedy_cpu_call(prompt, image_bytes, max_tokens=10):
    import ollama
    r = ollama.chat(
        model=llm_backend.OLLAMA_FALLBACK_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        keep_alive="30m",
        options={"num_ctx": 2048, "num_predict": max_tokens, "num_gpu": 0,
                 "temperature": 0, "top_k": 1, "seed": 0},
    )
    return r["message"]["content"].strip()


def greedy_vision_chat(prompt, image_bgr, max_tokens=10):
    image_bytes = llm_backend._encode_image(image_bgr)
    if image_bytes is None:
        return None

    def _gpu_once():
        try:
            t0 = time.time()
            out = _greedy_gpu_call(prompt, image_bytes, max_tokens)
            dt = time.time() - t0
        except Exception as e:
            print(f"[T0] GPU call raised ({e})")
            return None
        if llm_backend._is_degenerate(out):
            print(f"[T0] GPU degenerate ({dt:.1f}s)")
            return None
        print(f"[T0] GPU ok ({dt:.1f}s)")
        return out

    if llm_backend.ensure_running():
        for _ in range(2):
            out = _gpu_once()
            if out is not None:
                return out
        print("[T0] degenerate wedged -> restart")
        if llm_backend.restart():
            for _ in range(2):
                out = _gpu_once()
                if out is not None:
                    return out
    print("[T0] Falling back to CPU (greedy)")
    try:
        t0 = time.time()
        out = _greedy_cpu_call(prompt, image_bytes, max_tokens)
        print(f"[T0] CPU ok ({time.time()-t0:.1f}s)")
        return out
    except Exception as e:
        print(f"[T0] CPU fallback failed: {e}")
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


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    results = []
    n_frames = len([f for f in frames
                    if not FRAME_SUBSET or idx_of[f] in FRAME_SUBSET])
    print(f"DETERMINISTIC (temp=0) sweep: {n_frames} frames x "
          f"{len(PHRASINGS)} phrasings (subset={sorted(FRAME_SUBSET)})")
    for f in frames:
        idx = idx_of[f]
        if FRAME_SUBSET and idx not in FRAME_SUBSET:
            continue
        img = cv2.imread(f)
        gt = BOTTLE_GT[idx]
        for ph in PHRASINGS:
            t0 = time.time()
            ans = greedy_vision_chat(PROMPT.format(obj=ph), img, max_tokens=10)
            dt = round(time.time() - t0, 2)
            pred = parse(ans)
            rec = {"idx": idx, "file": f, "phrasing": ph,
                   "gt": gt, "raw": ans, "pred": pred, "latency_s": dt}
            results.append(rec)
            mark = "OK " if pred == gt else ("?? " if pred is None else "XX ")
            print(f"  [{idx:02d}] {ph:<24} gt={gt} pred={pred} {mark} "
                  f"{dt:5.1f}s  raw={ans!r}")
            with open("openvocab_phrasing_temp0_c2.json", "w") as fh:
                json.dump(results, fh, indent=2)
    llm_backend.shutdown()
    score(results)
    print("\nDONE -- wrote openvocab_phrasing_temp0_c2.json")


if __name__ == "__main__":
    main()
