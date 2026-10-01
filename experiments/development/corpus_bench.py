import json
import os
import sys

import cv2

DIRS = ["generated_corpus", "generated_occlusions"]
OUT = "corpus_bench.json"


def load_items():
    items = []
    seen = set()
    for d in DIRS:
        p = os.path.join(d, "manifest.json")
        if not os.path.exists(p):
            continue
        man = json.load(open(p))
        for it in man["items"]:
            key = os.path.basename(it["file"])
            if key in seen:
                continue
            seen.add(key)
            items.append({**it, "box": man["box"]})
    return items


def main():
    target = " ".join(sys.argv[1:]).strip() or "red cup"
    items = load_items()
    if not items:
        raise SystemExit("ABORT: no manifests -- run inpaint_corpus.py")

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
    print(f"{'item':<16}{'truth':>12}{'VLM says':>20}{'probe':>12}  result")
    print("-" * 72)
    for it in sorted(items, key=lambda x: (not x["is_occluder"], x["name"])):
        img = cv2.imread(it["file"])
        if img is None:
            continue
        pipe.box = it["box"]
        pipe._probe_at = None
        pipe._probe_verdict = None
        label, is_occ = pipe.probe_region(img)
        truth = bool(it["is_occluder"])
        said = bool(is_occ)
        if truth and said:
            res, kind = "ok", "hold_correct"
        elif truth and not said:
            res, kind = "false DROP (safe-ish)", "false_drop"
        elif not truth and said:
            res, kind = "FALSE HOLD (dangerous)", "false_hold"
        else:
            res, kind = "ok", "drop_correct"
        rows.append({"name": it["name"], "truth_occluder": truth,
                     "vlm_label": label, "probe_occluder": said,
                     "kind": kind, "file": it["file"]})
        print(f"{it['name']:<16}{'OCCLUDER' if truth else 'REPLACEMENT':>12}"
              f"{str(label):>20}{str(said):>12}  {res}")

    occ = [r for r in rows if r["truth_occluder"]]
    rep = [r for r in rows if not r["truth_occluder"]]
    hold_ok = sum(r["kind"] == "hold_correct" for r in rows)
    drop_ok = sum(r["kind"] == "drop_correct" for r in rows)
    fdrop = [r for r in rows if r["kind"] == "false_drop"]
    fhold = [r for r in rows if r["kind"] == "false_hold"]

    print("\n" + "=" * 72)
    print("CONFUSION MATRIX")
    print("=" * 72)
    print(f"{'':<22}{'probe: OCCLUDER':>18}{'probe: REPLACEMENT':>20}")
    print(f"{'truth OCCLUDER':<22}{hold_ok:>18}{len(fdrop):>20}")
    print(f"{'truth REPLACEMENT':<22}{len(fhold):>18}{drop_ok:>20}")
    print(f"\n  occluder recall   {hold_ok}/{len(occ)}   "
          f"(belief correctly HELD on a covered target)")
    print(f"  replacement recall {drop_ok}/{len(rep)}   "
          f"(belief correctly DROPPED on a gone target)")
    print(f"\n  false DROPS  {len(fdrop):<3} safe-ish -- lose a present target, "
          f"recoverable")
    if fdrop:
        print("      " + ", ".join(f"{r['name']}({r['vlm_label']})"
                                   for r in fdrop))
    print(f"  FALSE HOLDS  {len(fhold):<3} DANGEROUS -- belief held on a target "
          f"that is GONE")
    if fhold:
        print("      " + ", ".join(f"{r['name']}({r['vlm_label']})"
                                   for r in fhold))
        print("\n  Each false hold is the arm reaching for something that is "
              "not there.\n  These matter far more than the drops and must "
              "not be averaged with them.")

    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
