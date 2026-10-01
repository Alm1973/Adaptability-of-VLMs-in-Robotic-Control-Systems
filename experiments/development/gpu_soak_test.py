import glob
import statistics as st
import time

import cv2

import llm_backend
from openvocab_batched import PROMPT

N_CALLS = 40


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = [cv2.imread(f) for f in frames]
    print(f"soak: {N_CALLS} calls through llm_backend.vision_chat "
          f"(production path), cycling {len(imgs)} frames\n")

    lat = []
    degen = 0
    slow = 0
    for i in range(N_CALLS):
        img = imgs[i % len(imgs)]
        t0 = time.time()
        out = llm_backend.vision_chat(PROMPT, img, max_tokens=80)
        dt = time.time() - t0
        lat.append(dt)
        d = is_degenerate(out)
        degen += d
        if dt > 30:
            slow += 1
        print(f"  {i+1:2d}/{N_CALLS} {dt:6.1f}s "
              f"{'DEGEN' if d else 'ok   '} {str(out)[:45]!r}")

    llm_backend.shutdown()
    print("\n===== SOAK RESULT =====")
    print(f"degenerate answers : {degen}/{N_CALLS}   (must be 0)")
    print(f"slow (>30s) calls  : {slow}/{N_CALLS}   (CPU fallback; must be 0)")
    print(f"latency: mean={st.mean(lat):.1f}s median={st.median(lat):.1f}s "
          f"min={min(lat):.1f}s max={max(lat):.1f}s")
    print("VERDICT:", "PASS -- fix holds, all GPU"
          if degen == 0 and slow == 0 else "FAIL -- fix did not hold")


if __name__ == "__main__":
    main()
