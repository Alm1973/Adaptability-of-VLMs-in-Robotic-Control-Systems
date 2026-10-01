import glob
import json
import subprocess
import time

import cv2
import numpy as np

from backend_latency_ab import (WH, encode_b64, call, launch_b10326,
                                is_degenerate)

R = 4
N = 15


def vram_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def clean_gpu(min_free=3500, timeout=40):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    subprocess.run(["ollama", "stop", "qwen2.5vl:3b"], capture_output=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = vram_free_mib()
        if f >= min_free:
            return f
        time.sleep(1)
    return vram_free_mib()


def unique_images(seed, n):
    rng = np.random.default_rng(seed)
    bases = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]
    out = []
    for i in range(n):
        base = bases[i % len(bases)]
        h, w = base.shape[:2]
        cw, ch = int(w * 0.9), int(h * 0.9)
        x = int(rng.integers(0, w - cw + 1))
        y = int(rng.integers(0, h - ch + 1))
        img = cv2.resize(base[y:y + ch, x:x + cw], WH)
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=int(rng.integers(-25, 26)))
        out.append(encode_b64(img))
    return out


def main():
    runs = []
    for run in range(R):
        free = clean_gpu()
        print(f"\n### b10326 run {run+1}/{R}  (VRAM free {free} MiB before launch)")
        p = launch_b10326()
        if p is None:
            print("  launch FAILED even after VRAM-free wait")
            runs.append({"error": "launch failed", "vram_free": free})
            continue
        b64s = unique_images(seed=2000 + run, n=N)
        first_lock, degen, seq = None, 0, []
        for i, b64 in enumerate(b64s):
            try:
                out, _ = call(18081, b64)
            except Exception as e:
                seq.append("E")
                continue
            d = is_degenerate(out)
            seq.append("D" if d else ".")
            if d:
                degen += 1
                if first_lock is None:
                    first_lock = i + 1
        rec = {"first_lock": first_lock, "degen": degen, "n": len(seq),
               "seq": "".join(seq)}
        print(f"  {rec}")
        runs.append(rec)
    clean_gpu()

    print("\n============ b10326 SOLO STABILITY (480x360, unique imgs) ============")
    valid = [r for r in runs if "seq" in r]
    for i, r in enumerate(runs):
        print(f"  run {i+1}: {r}")
    clean_runs = sum(1 for r in valid if r["degen"] == 0)
    locks = [r["first_lock"] for r in valid if r["first_lock"]]
    print(f"\nvalid runs: {len(valid)}/{R}  clean(0 degen): {clean_runs}/{len(valid)}"
          f"  first_locks: {locks}  "
          f"total_degen: {sum(r['degen'] for r in valid)}/{sum(r['n'] for r in valid)}")
    json.dump(runs, open("b10326_stability_solo.json", "w"), indent=2)
    print("wrote b10326_stability_solo.json")


if __name__ == "__main__":
    main()
