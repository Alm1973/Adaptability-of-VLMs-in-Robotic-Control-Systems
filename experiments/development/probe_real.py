import json
import os
import re
import sys
from collections import Counter

import cv2

FRAMES = "live_disruption_frames"
OUT = "probe_real.json"
SAMPLES = 5

CASES = [
    ("occlusion_full", True, "a real hand"),
    ("occlude_book", True, "a real book"),
    ("occlude_paper", True, "real paper/cloth"),
    ("occlude_box", True, "a real cardboard box"),
    ("occlude_jacket", True, "a real jacket sleeve"),
]

GENERATED = {"occlusion_full": ("hand", True),
             "occlude_book": ("book", True),
             "occlude_paper": ("paper sheet", True),
             "occlude_box": ("bookend", False),
             "occlude_jacket": ("suit", False)}


def disrupt_frames(scenario):
    out = []
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_disrupt_(\d+)\.jpg$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(FRAMES, fn)))
    return [p for _, p in sorted(out)]


def baseline_box(scenario, target):
    import disruption_bench as db
    for fn in sorted(os.listdir(FRAMES)):
        if re.match(rf"^{re.escape(scenario)}_baseline_(\d+)\.jpg$", fn):
            got = db._target_box(cv2.imread(os.path.join(FRAMES, fn)), target)
            if got:
                return got[1]
    return None


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier

    v = make_fast_verifier()
    gen = getattr(v, "generate", None)
    if gen is None:
        raise SystemExit("ABORT: probe needs a GENERATIVE verifier")

    pipe = RecoveryPipeline(target, tracker=None, verifier=None,
                            use_detector=False, use_opencv=True,
                            use_vlm=False, use_state=True,
                            use_region_probe=True, probe_verifier=gen)

    rows = []
    print(f"{'scenario':<17}{'real labels (n=5)':<38}{'real':>7}"
          f"{'gen':>7}  agree")
    print("-" * 78)
    for scenario, truth, desc in CASES:
        paths = disrupt_frames(scenario)
        box = baseline_box(scenario, target)
        if not paths or box is None:
            print(f"{scenario:<17}(no frames or no baseline box)")
            continue
        pipe.box = box
        mid = paths[len(paths) // 3: len(paths) // 3 * 2] or paths
        step = max(1, len(mid) // SAMPLES)
        picks = mid[::step][:SAMPLES]

        labels, occ_votes = [], []
        for p in picks:
            img = cv2.imread(p)
            if img is None:
                continue
            pipe._probe_at = None
            pipe._probe_verdict = None
            lab, is_occ = pipe.probe_region(img)
            labels.append(str(lab))
            occ_votes.append(bool(is_occ))
        if not labels:
            continue
        real_occ = sum(occ_votes) > len(occ_votes) / 2
        gen_lab, gen_occ = GENERATED.get(scenario, (None, None))
        agree = (gen_occ is not None and real_occ == gen_occ)
        top = Counter(labels).most_common(3)
        shown = ", ".join(f"{l}x{c}" if c > 1 else l for l, c in top)
        rows.append({"scenario": scenario, "desc": desc, "truth": truth,
                     "real_labels": labels, "real_occluder": real_occ,
                     "real_votes": f"{sum(occ_votes)}/{len(occ_votes)}",
                     "generated_label": gen_lab,
                     "generated_occluder": gen_occ, "agree": agree})
        print(f"{scenario:<17}{shown[:36]:<38}{str(real_occ):>7}"
              f"{str(gen_occ):>7}  {'yes' if agree else 'NO'}")

    print("\n" + "=" * 78)
    print("REAL vs GENERATED OCCLUDERS")
    print("=" * 78)
    ok = [r for r in rows if r["real_occluder"] == r["truth"]]
    agree = [r for r in rows if r["agree"]]
    print(f"  probe correct on REAL occluders: {len(ok)}/{len(rows)}")
    print(f"  real and generated AGREE on:     {len(agree)}/{len(rows)}")
    disagree = [r for r in rows if not r["agree"]]
    if disagree:
        print("\n  DISAGREEMENTS (generated did not predict real):")
        for r in disagree:
            print(f"    {r['scenario']}: generated said "
                  f"{r['generated_label']!r} -> occluder={r['generated_occluder']}"
                  f"\n      real said {r['real_labels'][:3]} -> "
                  f"occluder={r['real_occluder']}")
        print("\n  Generated occluders do NOT predict real ones. That is the "
              "third data\n  category that fails to transfer, and it means the "
              "cheap CPU pre-filter\n  cannot replace a live session.")
    else:
        print("\n  Generated predicted real in every case. Generated occluders "
              "are usable\n  as a CHEAP PRE-FILTER for live sessions -- ~3 min "
              "of CPU instead of\n  booking a human at a desk. Still validate "
              "the final claim live.")

    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
