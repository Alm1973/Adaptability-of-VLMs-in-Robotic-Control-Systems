import statistics as st
import time

import cv2

from newbin_test import (NEW_EXE, balanced_accuracy, call, encode,
                         is_degenerate, kill, launch)
from openvocab_baseline import GT, PROMPT, VOCAB, parse


def main():
    frames = sorted(__import__("glob").glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}

    kill()
    print(f"launching FRESH {NEW_EXE}")
    p = launch("srv_newbin_cycle.log")
    if p is None:
        print("SERVER FAILED TO START. tail:")
        try:
            print("".join(open("srv_newbin_cycle.log").readlines()[-30:]))
        except Exception as e:
            print("(no log)", e)
        return
    print("server healthy\n")

    wh = (480, 360)
    rows, lat, degen = [], [], 0
    for f in frames:
        idx = idx_of[f]
        b64 = encode(imgs[f], wh)
        for obj in VOCAB:
            out, dt = call(b64, PROMPT.format(obj=obj))
            lat.append(dt)
            if is_degenerate(out):
                degen += 1
            rows.append({"idx": idx, "obj": obj,
                         "gt": GT[idx][obj], "pred": parse(out)})
        got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
        print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct  "
              f"(deg so far {degen})")
    kill()

    macro, fp_rate, per = balanced_accuracy(rows)
    print("\n===== b10326 @ 480x360 (standard Q4_K_M GGUF) =====")
    print(f"macro balanced acc : {macro:.3f}")
    print(f"FP rate            : {fp_rate:.3f}")
    print(f"degenerate finals  : {degen}/{len(rows)}")
    print(f"latency  med {st.median(lat):.2f}s  "
          f"p90 {sorted(lat)[int(0.9*len(lat))]:.2f}s  max {max(lat):.2f}s")
    print(f"per-object         : {per}")
    print("\nREFERENCE: shipped 480x360 macro 0.632-0.701 | "
          "Ollama-CPU@1024tok 0.889 (21-80s/call) | in-sample cycle-1 0.889")
    print("ADOPT only if macro is clearly above the shipped 0.632-0.701 band "
          "AND degenerate==0.")


if __name__ == "__main__":
    main()
