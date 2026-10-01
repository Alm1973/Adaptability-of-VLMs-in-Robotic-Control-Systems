import glob
import random
import statistics as st
import time

import cv2
import numpy as np

import llm_backend
from openvocab_baseline import PROMPT, VOCAB

N_CALLS = 60
RESOLUTION = (480, 360)


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def make_unique(img, rng, nprng):
    h, w = img.shape[:2]
    dx, dy = rng.randint(0, 40), rng.randint(0, 30)
    cw, ch = w - rng.randint(0, 40), h - rng.randint(0, 30)
    crop = img[dy:dy + ch, dx:dx + cw].copy()
    crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=rng.randint(-18, 18))
    crop[0:4, 0:4] = nprng.randint(0, 255, size=(4, 4, 3)).astype(np.uint8)
    return crop


def main():
    rng, nprng = random.Random(99), np.random.RandomState(99)
    srcs = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]
    if not srcs:
        print("no exp_*.jpg found")
        return

    print(f"ROTATION VALIDATION: {N_CALLS} calls through "
          f"llm_backend.vision_chat (PRODUCTION path)")
    print(f"every image UNIQUE; ENCODE_BUDGET={llm_backend.ENCODE_BUDGET} "
          f"-> expect ~{N_CALLS // llm_backend.ENCODE_BUDGET} proactive "
          f"restarts, 0 corruption\n")

    lat, degen_at = [], []
    wall0 = time.time()
    for i in range(N_CALLS):
        img = make_unique(srcs[i % len(srcs)], rng, nprng)
        obj = VOCAB[i % len(VOCAB)]
        t0 = time.time()
        out = llm_backend.vision_chat(PROMPT.format(obj=obj), img,
                                      max_tokens=20)
        dt = time.time() - t0
        lat.append(dt)
        d = is_degenerate(out)
        if d:
            degen_at.append(i + 1)
        print(f"  {i+1:3d}/{N_CALLS} {dt:6.2f}s "
              f"{'DEGEN' if d else 'ok   '} {obj:<14} {str(out)[:20]!r}")

    wall = time.time() - wall0
    llm_backend.shutdown()

    slow = [x for x in lat if x > 30]
    print("\n===== ROTATION VALIDATION RESULT =====")
    print(f"calls (all unique) : {N_CALLS}")
    print(f"degenerate         : {len(degen_at)}   (must be 0)")
    if degen_at:
        print(f"  degenerate at    : {degen_at[:20]}")
    print(f"CPU fallbacks(>30s): {len(slow)}")
    print(f"latency p50 {sorted(lat)[len(lat)//2]:.2f}s  "
          f"mean {st.mean(lat):.2f}s  max {max(lat):.2f}s")
    print(f"wall total {wall:.1f}s -> amortized {wall/N_CALLS:.2f}s/query "
          f"(includes proactive restarts)")
    ok = len(degen_at) == 0
    print("VERDICT:", "PASS -- no corruption under sustained unique-image load"
          if ok else "FAIL -- rotation did not prevent corruption")


if __name__ == "__main__":
    main()
