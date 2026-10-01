import glob
import statistics as st
import time

import cv2

import llm_backend
from adaptive_detector import configure, detect_candidates, verify

QUERIES = ["water bottle", "shiniest object", "round object", "biggest object"]


def main():
    frames = sorted(glob.glob("exp_*.jpg"))[:4]
    imgs = {f: cv2.imread(f) for f in frames}
    ref = frames[0]
    print(f"CUE PIPELINE TEST  frames={len(frames)}  placement=GPU\n")

    for q in QUERIES:
        print("=" * 62)
        print(f"QUERY: {q!r}")
        cfg, cdt = configure(imgs[ref], q)
        if cfg is None:
            print("  could not configure (VLM did not locate it)\n")
            continue
        print(f"  cues       : {cfg.cues}   superlative={cfg.superlative}")
        print(f"  colour     : {cfg.color_name!r}  cell={cfg.cell}")
        print(f"  notes      : {cfg.notes}")
        print(f"  config cost: {cdt:.2f}s")

        cv_times = []
        for f in frames:
            t0 = time.time()
            cands = detect_candidates(imgs[f], cfg)
            cv_times.append(time.time() - t0)
            top = cands[0] if cands else None
            if top:
                feats = top.get("features", {})
                bits = []
                for k in ("circularity", "specular_frac", "mean_v", "area"):
                    if k in feats:
                        bits.append(f"{k}={feats[k]:.3f}")
                print(f"    {f[:22]:24} cands={len(cands):2d} "
                      f"top_area={top['area']:.0f} "
                      f"score={top.get('score', float('nan')):.2f} "
                      f"{' '.join(bits)}")
            else:
                print(f"    {f[:22]:24} cands= 0")
        print(f"  opencv per frame: {st.mean(cv_times)*1000:.1f} ms\n")

    llm_backend.shutdown()


if __name__ == "__main__":
    main()
