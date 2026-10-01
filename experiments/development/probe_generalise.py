import json
import os

import cv2

from recovery_pipeline import RecoveryPipeline

INDIR = "generated_occlusions"
OUT = "probe_generalise.json"


def main():
    man_path = os.path.join(INDIR, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(f"ABORT: no {man_path} -- run inpaint_occluders.py")
    man = json.load(open(man_path))
    box = man["box"]

    from run_study import make_fast_verifier
    verifier = make_fast_verifier()
    probe_v = getattr(verifier, "generate", None)
    if probe_v is None:
        raise SystemExit("ABORT: verifier has no .generate -- the probe needs "
                         "a GENERATIVE model. The fast logits path cannot "
                         "answer an open question; it once scored a hand 0/10 "
                         "and 'disproved' the probe entirely.")

    pipe = RecoveryPipeline("red cup", tracker=None, verifier=None,
                            use_detector=False, use_opencv=True,
                            use_vlm=False, use_state=True,
                            use_region_probe=True, probe_verifier=probe_v)
    pipe.box = box

    rows = []
    print(f"OCCLUDER_WORDS = {pipe.OCCLUDER_WORDS}\n")
    print(f"{'occluder':<10}{'truth':>9}{'VLM says':>18}{'in-list':>9}"
          f"   verdict")
    print("-" * 62)
    for item in man["items"]:
        img = cv2.imread(item["file"])
        if img is None:
            continue
        label, is_occ = pipe.probe_region(img)
        truth = item["is_occluder"]
        ok = (bool(is_occ) == truth)
        rows.append({"name": item["name"], "truth_occluder": truth,
                     "vlm_label": label, "classified_occluder": bool(is_occ),
                     "correct": ok, "file": item["file"]})
        print(f"{item['name']:<10}{'OCCLUDER' if truth else 'replace':>9}"
              f"{str(label):>18}{str(bool(is_occ)):>9}   "
              f"{'ok' if ok else 'WRONG -> belief dropped with cup present'}")

    print("\n" + "=" * 62)
    print("DOES THE PROBE GENERALISE PAST A HAND?")
    print("=" * 62)
    bad = [r for r in rows if not r["correct"]]
    print(f"  {len(rows) - len(bad)}/{len(rows)} classified correctly")
    if bad:
        print(f"  FAILED: {', '.join(r['name'] for r in bad)}")
        for r in bad:
            print(f"    {r['name']}: VLM said {r['vlm_label']!r}, which is not "
                  f"in OCCLUDER_WORDS,\n      so a genuine occluder reads as a "
                  f"replacement and belief is DROPPED\n      while the cup is "
                  f"still behind it.")
        print("\n  This is the keyword-list brittleness, demonstrated. The fix "
              "is NOT a\n  longer list -- the next unlisted noun fails "
              "identically. It needs the\n  question answered as a "
              "CLASSIFICATION, not matched as a string.\n  (A constrained "
              "two-way prompt was already tried and measured WORSE:\n  27/30 "
              "vs 28/30, with its extra errors in the dangerous direction.)")
    else:
        print("  All classified correctly. NOTE this is weak evidence: "
              "rendered\n  occluders may simply be easier to name than real "
              "ones. It does NOT\n  retire occlude_book / occlude_paper.")

    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
