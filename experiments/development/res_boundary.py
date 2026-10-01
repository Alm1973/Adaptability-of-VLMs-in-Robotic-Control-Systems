import glob
import statistics as st
import time

import cv2

import llm_backend
from openvocab_baseline import GT, PROMPT, VOCAB, parse

CANDIDATES = [(480, 360), (512, 384), (544, 408), (576, 432), (608, 456)]
N_SOAK = 10
ACC_MIN_STABLE = True


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def balanced_accuracy(rows):
    bal, per = [], {}
    all_neg = all_fp = 0
    for obj in VOCAB:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos, nneg = tp + fn, tn + fp
        all_neg += nneg
        all_fp += fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        if sens is not None and spec is not None:
            bal.append((sens + spec) / 2)
            per[obj] = round((sens + spec) / 2, 3)
        else:
            per[obj] = None
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, per)


def patched_encode(wh):
    import cv2 as _cv2

    def _enc(image_bgr):
        small = _cv2.resize(image_bgr, wh)
        ok, buf = _cv2.imencode(".jpg", small)
        return buf.tobytes() if ok else None
    return _enc


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    orig_encode = llm_backend._encode_image

    print("resolution boundary search on the PRODUCTION backend")
    print(f"soak {N_SOAK} calls/size, then full accuracy at the winner\n")

    results = {}
    best = None
    for wh in CANDIDATES:
        label = f"{wh[0]}x{wh[1]}"
        llm_backend._encode_image = patched_encode(wh)
        llm_backend.shutdown()
        time.sleep(1)
        print(f"##### {label} : soak #####")
        deg, lat = 0, []
        for i in range(N_SOAK):
            f = frames[i % len(frames)]
            t0 = time.time()
            out = llm_backend.vision_chat(PROMPT.format(obj="keyboard"),
                                          imgs[f], max_tokens=20)
            dt = time.time() - t0
            lat.append(dt)
            d = is_degenerate(out)
            deg += d
            print(f"  {i+1:2d}/{N_SOAK} {dt:5.1f}s "
                  f"{'DEGEN' if d else 'ok   '} {str(out)[:30]!r}")
        med = round(st.median(lat), 2)
        results[label] = {"soak_degen": deg, "n_soak": N_SOAK,
                          "med_lat_s": med}
        if deg:
            print(f"  -> {deg}/{N_SOAK} degenerate: UNSTABLE, boundary found\n")
            break
        print(f"  -> {N_SOAK}/{N_SOAK} clean, med {med}s: stable\n")
        best = wh

    if best is None:
        print("no stable resolution found (unexpected)")
        llm_backend._encode_image = orig_encode
        llm_backend.shutdown()
        return

    label = f"{best[0]}x{best[1]}"
    print(f"##### FULL ACCURACY at largest stable size: {label} #####")
    llm_backend._encode_image = patched_encode(best)
    llm_backend.shutdown()
    time.sleep(1)
    rows, alat = [], []
    deg = 0
    for f in frames:
        idx = idx_of[f]
        for obj in VOCAB:
            t0 = time.time()
            out = llm_backend.vision_chat(PROMPT.format(obj=obj),
                                          imgs[f], max_tokens=20)
            alat.append(time.time() - t0)
            if is_degenerate(out):
                deg += 1
            rows.append({"idx": idx, "obj": obj,
                         "gt": GT[idx][obj], "pred": parse(out)})
        got = sum(1 for r in rows[-len(VOCAB):] if r["pred"] == r["gt"])
        print(f"  [{idx:02d}] {got}/{len(VOCAB)} correct")

    macro, fp_rate, per = balanced_accuracy(rows)
    results[label].update({
        "macro_bal_acc": round(macro, 3) if macro else None,
        "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "acc_degen": deg, "per_object": per,
        "med_acc_lat_s": round(st.median(alat), 2),
        "total_lat_s": round(sum(alat), 1),
    })
    import json
    json.dump(results, open("res_boundary.json", "w"), indent=2)

    llm_backend._encode_image = orig_encode
    llm_backend.shutdown()

    print("\n===== RESOLUTION BOUNDARY SUMMARY =====")
    for lb, r in results.items():
        print(f"{lb:<10} soak {r['soak_degen']}/{r['n_soak']} degen  "
              f"med {r['med_lat_s']}s  macro {r.get('macro_bal_acc','-')}")
    s = results[label]
    print(f"\nSHIPPABLE: {label} -> macro {s.get('macro_bal_acc')} "
          f"FP {s.get('fp_rate')} @ {s.get('med_acc_lat_s')}s/query, "
          f"{s['soak_degen']}/{s['n_soak']} degenerate")
    print(f"per-object: {s.get('per_object')}")
    print("\nRefs: 480x360 macro 0.632 @2.5s | CPU macro 0.889 @21-80s")
    print(f"To ship: set _encode_image resize to {best} in llm_backend.py")


if __name__ == "__main__":
    main()
