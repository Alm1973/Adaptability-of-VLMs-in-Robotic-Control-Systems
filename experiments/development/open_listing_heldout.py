import glob
import json
import re
import statistics as st
import time

import cv2

import llm_backend
from open_listing_test import PROMPT, SYNONYMS, in_list, parse_items

MAX_ITEMS = 25

HALLUCINATION_PROBES = ["banana", "bicycle", "elephant", "umbrella"]
PROBE_SYNONYMS = {
    "banana": ["banana"],
    "bicycle": ["bicycle", "bike"],
    "elephant": ["elephant"],
    "umbrella": ["umbrella"],
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
        for syn in PROBE_SYNONYMS[probe]:
            if re.search(rf"\b{re.escape(syn)}\b", s):
                return True
    return False


def main():
    frames = sorted(glob.glob("c2_*.jpg"))
    if not frames:
        print("no c2_*.jpg found")
        return
    print(f"OPEN-LISTING HELD-OUT VALIDATION on {len(frames)} c2_* frames "
          f"(1 call each)\n")

    lists, lat = {}, []
    for f in frames:
        img = cv2.imread(f)
        t0 = time.time()
        out = llm_backend.vision_chat(PROMPT, img, max_tokens=100)
        dt = time.time() - t0
        lat.append(dt)
        items = dedupe(parse_items(out))
        lists[f] = items
        print(f"  {f:<26} {dt:5.1f}s  {items}")

    print("\n=== HALLUCINATION PROBES (should never appear) ===")
    hits, checks = 0, 0
    detail = []
    for f in frames:
        found = [p for p in HALLUCINATION_PROBES if probe_hit(p, lists[f])]
        checks += len(HALLUCINATION_PROBES)
        hits += len(found)
        if found:
            detail.append((f, found))
        print(f"  {f:<26} {found if found else 'clean'}")
    halluc_rate = hits / checks if checks else None

    print("\n=== VERIFIED POSITIVES (labeled by eye) ===")
    correct = total = 0
    misses = []
    for f, labels in VERIFIED_POSITIVES.items():
        if f not in lists:
            continue
        for obj, gt in labels.items():
            pred = 1 if in_list(obj, lists[f]) else 0
            total += 1
            ok = pred == gt
            correct += ok
            if not ok:
                misses.append((f, obj, gt, pred))
            print(f"  {f:<26} {obj:<14} gt={gt} pred={pred} "
                  f"{'OK' if ok else 'XX'}")

    llm_backend.shutdown()
    res = {
        "n_frames": len(frames),
        "halluc_hits": hits, "halluc_checks": checks,
        "halluc_rate": round(halluc_rate, 3) if halluc_rate is not None else None,
        "halluc_detail": detail,
        "pos_correct": correct, "pos_total": total, "pos_misses": misses,
        "med_lat_s": round(st.median(lat), 2),
        "lists": lists,
    }
    json.dump(res, open("open_listing_heldout.json", "w"), indent=2)

    print("\n===== HELD-OUT SUMMARY (open listing) =====")
    print(f"hallucination rate : {res['halluc_rate']} ({hits}/{checks})")
    print(f"verified positives : {correct}/{total}"
          + (f"  ({correct/total:.2f})" if total else ""))
    print(f"median latency     : {res['med_lat_s']}s per frame "
          f"(covers ALL objects)")
    print("\nPer-object prompt on the SAME held-out frames scored:")
    print("  negative-control FP 0.15 (6/40), verified positives 4/8 (0.50)")
    print("  and hallucinated bicycle on 3 frames + banana on 2.")


if __name__ == "__main__":
    main()
