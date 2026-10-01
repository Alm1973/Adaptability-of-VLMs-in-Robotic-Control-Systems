import glob
import json
import re
import statistics as st
import time

import cv2

import llm_backend
from open_listing_test import in_list, parse_items

MAX_ITEMS = 25

PROMPTS = {
    "current": ("List every distinct physical object you can clearly see in "
                "this image. Reply with ONLY a comma-separated list of short "
                "object names, nothing else."),
    "count_hint": ("Carefully scan this whole image and list at least 8 "
                   "different physical objects you can see, from the largest "
                   "to the smallest. Include partially visible objects. Reply "
                   "with ONLY a comma-separated list of short object names."),
}

HALLUCINATION_PROBES = {
    "banana": ["banana"], "bicycle": ["bicycle", "bike"],
    "elephant": ["elephant"], "umbrella": ["umbrella"],
}

VERIFIED_POSITIVES = {
    "c2_07_tilt_b90_t78.jpg": {"laptop": 1, "water bottle": 1,
                               "keyboard": 0, "computer mouse": 0},
    "c2_10_home.jpg":         {"laptop": 1, "water bottle": 1,
                               "keyboard": 1, "computer mouse": 1},
}


def dedupe(items):
    seen, out = set(), []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= MAX_ITEMS:
            break
    return out


def probe_hit(probe, items):
    for raw in items:
        s = raw.strip().lower()
        for syn in HALLUCINATION_PROBES[probe]:
            if re.search(rf"\b{re.escape(syn)}\b", s):
                return True
    return False


def main():
    frames = sorted(glob.glob("c2_*.jpg"))
    if not frames:
        print("no c2_*.jpg found")
        return
    imgs = {f: cv2.imread(f) for f in frames}
    print(f"HELD-OUT FINAL: {len(PROMPTS)} finalists x {len(frames)} frames\n")

    summary = {}
    for name, prompt in PROMPTS.items():
        print(f"##### {name} #####")
        lists, lat = {}, []
        for f in frames:
            t0 = time.time()
            out = llm_backend.vision_chat(prompt, imgs[f], max_tokens=140)
            dt = time.time() - t0
            lat.append(dt)
            lists[f] = dedupe(parse_items(out))
            print(f"  {f:<26} {dt:5.1f}s ({len(lists[f]):2d}) {lists[f][:8]}")

        hits = checks = 0
        detail = []
        for f in frames:
            found = [p for p in HALLUCINATION_PROBES if probe_hit(p, lists[f])]
            checks += len(HALLUCINATION_PROBES)
            hits += len(found)
            if found:
                detail.append((f, found))

        correct = total = 0
        misses = []
        for f, labels in VERIFIED_POSITIVES.items():
            if f not in lists:
                continue
            for obj, gt in labels.items():
                pred = 1 if in_list(obj, lists[f]) else 0
                total += 1
                if pred == gt:
                    correct += 1
                else:
                    misses.append((f, obj, gt, pred))

        summary[name] = {
            "halluc_hits": hits, "halluc_checks": checks,
            "halluc_rate": round(hits / checks, 3) if checks else None,
            "halluc_detail": detail,
            "pos_correct": correct, "pos_total": total, "pos_misses": misses,
            "mean_items": round(st.mean([len(v) for v in lists.values()]), 1),
            "med_lat_s": round(st.median(lat), 2),
            "lists": lists,
        }
        json.dump(summary, open("listing_heldout_final.json", "w"), indent=2)
        s = summary[name]
        print(f"  -> halluc {hits}/{checks} ({s['halluc_rate']})  "
              f"positives {correct}/{total}  mean_items {s['mean_items']}  "
              f"med {s['med_lat_s']}s\n")

    llm_backend.shutdown()
    print("===== HELD-OUT FINAL SUMMARY =====")
    print(f"{'variant':<12} {'halluc':<12} {'positives':<12} "
          f"{'items':<8} {'med_s'}")
    for n, s in summary.items():
        print(f"{n:<12} {s['halluc_hits']}/{s['halluc_checks']} "
              f"({s['halluc_rate']})".ljust(12)
              + f"  {s['pos_correct']}/{s['pos_total']}".ljust(12)
              + f"  {s['mean_items']:<8} {s['med_lat_s']}")
    print("\nReference on the SAME held-out frames:")
    print("  per-object prompt: hallucinated 6/40 (bicycle x3, banana x2), "
          "positives 4/8, ~2.7s PER OBJECT")
    for n, s in summary.items():
        if s["halluc_detail"]:
            print(f"  {n} hallucinated: {s['halluc_detail']}")
        if s["pos_misses"]:
            print(f"  {n} missed: {[(m[0][:10], m[1]) for m in s['pos_misses']]}")


if __name__ == "__main__":
    main()
