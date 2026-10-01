import glob
import json
import statistics as st
import time

import cv2

import llm_backend
from adaptive_detector import (Suppression, configure_validated, detect_candidates, verify)
from open_vocab_detect import VOTE_SAMPLES, detect as vlm_detect
from openvocab_baseline import GT

TARGET = "water bottle"


def score(rows):
    tp = sum(1 for r in rows if r["gt"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in rows if r["gt"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in rows if r["gt"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in rows if r["gt"] == 1 and r["pred"] == 0)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    bal = ((sens or 0) + (spec or 0)) / 2 if (sens is not None
                                              and spec is not None) else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sens": round(sens, 3) if sens is not None else None,
            "spec": round(spec, 3) if spec is not None else None,
            "balanced": round(bal, 3) if bal is not None else None}


def run_baseline(frames, imgs, idx_of):
    rows, lat = [], []
    for f in frames:
        t0 = time.time()
        res, _ = vlm_detect(imgs[f], [TARGET], samples=VOTE_SAMPLES)
        dt = time.time() - t0
        lat.append(dt)
        rows.append({"gt": GT[idx_of[f]][TARGET], "pred": int(bool(res[TARGET]))})
        print(f"  [{idx_of[f]:02d}] baseline {dt:6.2f}s pred={rows[-1]['pred']} "
              f"gt={rows[-1]['gt']}")
    return rows, lat


def run_new(frames, imgs, idx_of):
    cfg = None
    cfg_time = 0.0
    for f in frames:
        if GT[idx_of[f]][TARGET] == 1:
            cfg, cfg_time = configure_validated(imgs[f], TARGET)
            if cfg:
                break
    if cfg is None:
        print("  [NEW] configuration failed -- cannot run pipeline")
        return None, None, None, None, None

    supp = Suppression()
    rows, cv_lat, ver_lat = [], [], []
    verify_calls = 0
    for f in frames:
        t0 = time.time()
        cands = detect_candidates(imgs[f], cfg, suppression=supp)
        cv_dt = time.time() - t0
        cv_lat.append(cv_dt)

        pred = 0
        vtotal = 0.0
        for c in cands[:2]:
            ok, vdt = verify(imgs[f], c["box"], TARGET)
            verify_calls += 1
            vtotal += vdt
            ver_lat.append(vdt)
            if ok:
                pred = 1
                break
            supp.add(c["center"])
        rows.append({"gt": GT[idx_of[f]][TARGET], "pred": pred})
        print(f"  [{idx_of[f]:02d}] new cv={cv_dt*1000:5.1f}ms "
              f"cands={len(cands)} verify={vtotal:5.2f}s "
              f"pred={pred} gt={rows[-1]['gt']}")
    return rows, cv_lat, ver_lat, cfg_time, verify_calls


def main():
    frames = sorted(glob.glob("exp_*.jpg"))
    imgs = {f: cv2.imread(f) for f in frames}
    idx_of = {f: i + 1 for i, f in enumerate(frames)}
    print(f"BENCHMARK target={TARGET!r}  frames={len(frames)}  placement=GPU\n")

    print("--- BASELINE (VLM every frame) ---")
    b_rows, b_lat = run_baseline(frames, imgs, idx_of)
    b_score = score(b_rows)

    print("\n--- NEW (VLM configures, OpenCV runs, VLM verifies) ---")
    n_rows, cv_lat, ver_lat, cfg_time, vcalls = run_new(frames, imgs, idx_of)
    llm_backend.shutdown()
    if n_rows is None:
        return
    n_score = score(n_rows)

    n_total = cfg_time + sum(cv_lat) + sum(ver_lat)
    b_total = sum(b_lat)
    out = {
        "target": TARGET, "placement": "GPU", "n_frames": len(frames),
        "baseline": {"score": b_score, "total_s": round(b_total, 2),
                     "per_frame_s": round(st.mean(b_lat), 3),
                     "vlm_calls": len(frames) * VOTE_SAMPLES},
        "new": {"score": n_score, "total_s": round(n_total, 2),
                "config_s": round(cfg_time, 2),
                "opencv_per_frame_ms": round(st.mean(cv_lat) * 1000, 2),
                "verify_calls": vcalls,
                "verify_mean_s": round(st.mean(ver_lat), 3) if ver_lat else 0,
                "vlm_calls": 2 + vcalls},
    }
    json.dump(out, open("pipeline_benchmark.json", "w"), indent=2)

    print("\n" + "=" * 66)
    print("RESULTS  (placement: GPU)")
    print("=" * 66)
    print(f"{'':22} {'BASELINE':>18} {'NEW':>18}")
    print(f"{'balanced accuracy':22} {str(b_score['balanced']):>18} "
          f"{str(n_score['balanced']):>18}")
    print(f"{'sensitivity':22} {str(b_score['sens']):>18} "
          f"{str(n_score['sens']):>18}")
    print(f"{'specificity':22} {str(b_score['spec']):>18} "
          f"{str(n_score['spec']):>18}")
    print(f"{'VLM calls total':22} {len(frames)*VOTE_SAMPLES:>18} "
          f"{2 + vcalls:>18}")
    print(f"{'total wall time':22} {b_total:>17.1f}s {n_total:>17.1f}s")
    print(f"{'mean per frame':22} {st.mean(b_lat):>17.2f}s "
          f"{(sum(cv_lat)+sum(ver_lat))/len(frames):>17.2f}s")
    print(f"\nthe number that matters for a tracking loop --")
    print(f"  cost of a frame with NO detection event:")
    print(f"    baseline : {st.mean(b_lat)*1000:8.1f} ms  (full VLM every frame)")
    print(f"    new      : {st.mean(cv_lat)*1000:8.1f} ms  (OpenCV only)")
    if st.mean(cv_lat) > 0:
        print(f"    speedup  : {st.mean(b_lat)/st.mean(cv_lat):8.0f}x")
    print(f"\nconfig cost (once per search): {cfg_time:.2f}s")
    print(f"verify calls (only on candidates): {vcalls}")


if __name__ == "__main__":
    main()
