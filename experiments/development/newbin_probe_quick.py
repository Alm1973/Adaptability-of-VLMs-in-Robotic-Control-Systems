import base64
import glob
import random

import cv2
import numpy as np

from newbin_test import NEW_EXE, is_degenerate, kill, launch, call

PROMPT = "Is there a keyboard in this image? Answer with only one word: yes or no."


def gpu_confirmed(logpath):
    try:
        txt = open(logpath, encoding="utf-8", errors="ignore").read()
    except Exception:
        return False, "(no log)"
    offloaded = "offloaded" in txt and "layers to GPU" in txt
    clip_cuda = "CLIP using CUDA" in txt
    clip_cpu = "CLIP using CPU" in txt
    return (offloaded and clip_cuda and not clip_cpu,
            f"offloaded={offloaded} clip_cuda={clip_cuda} clip_cpu={clip_cpu}")


def unique_encode(img, rng, wh):
    h, w = img.shape[:2]
    dx, dy = rng.randint(0, 40), rng.randint(0, 30)
    cw, ch = w - rng.randint(0, 40), h - rng.randint(0, 30)
    crop = img[dy:dy + ch, dx:dx + cw]
    crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=rng.randint(-18, 18))
    crop[0:4, 0:4] = np.random.randint(0, 255, size=(4, 4, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, wh))
    return base64.b64encode(buf.tobytes()).decode()


def main():
    rng = random.Random(7)
    srcs = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]

    kill()
    print(f"launching {NEW_EXE}")
    p = launch("srv_newbin_probe.log")
    if p is None:
        print("SERVER FAILED TO START. log tail:")
        try:
            print("".join(open("srv_newbin_probe.log").readlines()[-30:]))
        except Exception as e:
            print("(no log)", e)
        return
    ok, why = gpu_confirmed("srv_newbin_probe.log")
    print(f"GPU offload check: {'OK' if ok else 'WARNING'} -- {why}")

    wh = (640, 480)
    print(f"\n5 UNIQUE-image calls at {wh[0]}x{wh[1]} "
          f"(old binary locked by ~call 3):")
    degen = 0
    for i in range(5):
        out, dt = call(unique_encode(srcs[i % len(srcs)], rng, wh), PROMPT)
        d = is_degenerate(out)
        degen += d
        print(f"  {i+1}/5  {dt:5.1f}s  {'DEGEN' if d else 'ok   '}  {out[:40]!r}")
    kill()
    print(f"\n-> {degen}/5 degenerate at {wh[0]}x{wh[1]}. "
          + ("HYPOTHESIS DEAD (locks like old binary)"
             if degen else "CLEAN -- proceed to full cycle"))


if __name__ == "__main__":
    main()
