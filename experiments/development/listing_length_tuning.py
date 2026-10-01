import glob
import json
import re
import statistics as st
import time

import cv2

import llm_backend
from open_vocab_detect import _dedupe, _parse_items, is_present
from openvocab_baseline import GT, VOCAB

VARIANTS = {
    "shipped": ("List every distinct physical object you can clearly see in "
                "this image. Reply with ONLY a comma-separated list of short "
                "object names, nothing else."),
    "atleast4": ("List the physical objects you can clearly see in this "
                 "image, at least 4 of them if that many are present. Reply "
                 "with ONLY a comma-separated list of short object names."),
    "atleast5_nofill": ("List the physical objects you can clearly see in "
                        "this image. Try to name at least 5, but ONLY include "
                        "objects you can actually see - never guess or add "
                        "objects that might typically be there. Reply with "
                        "ONLY a comma-separated list of short object names."),
    "dont_stop": ("List the physical objects you can clearly see in this "
                  "image. Do not stop after the most obvious ones - also "
                  "name smaller objects that are actually visible, but do "
                  "not invent anything. Reply with ONLY a comma-separated "
                  "list of short object names."),
}

HALLUC = {"banana": ["banana"], "bicycle": ["bicycle", "bike"],
          "elephant": ["elephant"], "umbrella": ["umbrella"]}
VERIFIED = {
    "c2_07_tilt_b90_t78.jpg": {"laptop": 1, "water bottle": 1,
                               "keyboard": 0, "computer mouse": 0},
    "c2_10_home.jpg":         {"laptop": 1, "water bottle": 1,
                               "keyboard": 1, "computer mouse": 1},
}


def probe_hit(p, items):
    return any(re.search(rf"\b{re.escape(s)}\b", it.strip().lower())
               for it in items for s in HALLUC[p])


def score_insample(rows):
    bal, sens_map = [], {}
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
        sens_map[obj] = round(sens, 3) if sens is not None else None
        if sens is not None and spec is not None:
            bal.append((sens + spec) / 2)
    return (st.mean(bal) if bal else None,
            all_fp / all_neg if all_neg else None, sens_map)


def run_prompt(prompt, paths, imgs, max_tokens=140):
    out = {}
    lat = []
    for f in paths:
        t0 = time.time()
        txt = llm_backend.vision_chat(prompt, imgs[f], max_tokens=max_tokens)
        lat.append(time.time() - t0)
        out[f] = _dedupe(_parse_items(txt))
    return out, lat


def main():
    exp = sorted(glob.glob("exp_*.jpg"))
    c2 = sorted(glob.glob("c2_*.jpg"))
    imgs = {f: cv2.imread(f) for f in exp + c2}
    idx_of = {f: i + 1 for i, f in enumerate(exp)}
    print(f"LENGTH TUNING: {len(VARIANTS)} variants | in-sample {len(exp)} "
          f"frames + held-out {len(c2)} frames\n")

    summary = {}
    for name, prompt in VARIANTS.items():
        print(f"##### {name} #####")
        lists, lat = run_prompt(prompt, exp, imgs)
        rows = []
        for f in exp:
            idx = idx_of[f]
            for obj in VOCAB:
                rows.append({"obj": obj, "gt": GT[idx][obj],
                             "pred": 1 if is_present(obj, lists[f]) else 0})
        macro, fp, sens = score_insample(rows)
        n_in = st.mean([len(v) for v in lists.values()])

        hlists, hlat = run_prompt(prompt, c2, imgs)
        hits = checks = 0
        for f in c2:
            for p in HALLUC:
                checks += 1
                hits += probe_hit(p, hlists[f])
        correct = total = 0
        misses = []
        for f, labels in VERIFIED.items():
            for obj, gt in labels.items():
                pred = 1 if is_present(obj, hlists.get(f, [])) else 0
                total += 1
                if pred == gt:
                    correct += 1
                else:
                    misses.append((f[:10], obj, gt, pred))
        n_out = st.mean([len(v) for v in hlists.values()])

        summary[name] = {
            "in_macro": round(macro, 3) if macro else None,
            "in_fp": round(fp, 3) if fp is not None else None,
            "in_bottle_sens": sens.get("water bottle"),
            "in_items": round(n_in, 1),
            "held_halluc": f"{hits}/{checks}",
            "held_positives": f"{correct}/{total}",
            "held_items": round(n_out, 1),
            "held_misses": misses,
            "med_lat_s": round(st.median(lat + hlat), 2),
        }
        json.dump(summary, open("listing_length_tuning.json", "w"), indent=2)
        s = summary[name]
        print(f"  in-sample : macro {s['in_macro']} FP {s['in_fp']} "
              f"bottle_sens {s['in_bottle_sens']} items {s['in_items']}")
        print(f"  HELD-OUT  : halluc {s['held_halluc']} positives "
              f"{s['held_positives']} items {s['held_items']}")
        print(f"  latency   : {s['med_lat_s']}s\n")

    llm_backend.shutdown()
    print("===== LENGTH TUNING SUMMARY =====")
    print(f"{'variant':<18} {'in_macro':<9} {'bottle':<8} {'HELD_pos':<10} "
          f"{'HELD_hal':<10} {'items':<7} {'med_s'}")
    for n, s in summary.items():
        print(f"{n:<18} {str(s['in_macro']):<9} "
              f"{str(s['in_bottle_sens']):<8} {s['held_positives']:<10} "
              f"{s['held_halluc']:<10} {str(s['held_items']):<7} "
              f"{s['med_lat_s']}")
    print("\nSHIP RULE: beat held-out 6/8 positives AND keep held-out "
          "hallucinations 0/40. In-sample macro is NOT the deciding number "
          "(count_hint won in-sample 0.868 and lost held-out 5/8).")


if __name__ == "__main__":
    main()
