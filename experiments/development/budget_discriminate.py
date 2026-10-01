import gc
import json
import time

import cv2

import disruption_bench as db

BUDGETS = [100352, 200704]
SUBTLE_HUES = (20, 30, 45)
PADS = (0.25, 1.5, 3.0)
OUT = "budget_discriminate.json"
WARMUP = 2
PROMPT = "Answer yes or no only. Is the main object in this image a {}?"


def crop_at(img, box, pad_ratio):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    c = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    return c if c.size else None


def build(frame, box, target):
    out = []
    gone = db._remove_target(frame, box)

    for pad in PADS:
        tag = f"pad{pad}"
        c = crop_at(frame, box, pad)
        if c is not None:
            out.append((c, True, f"target {tag}"))
        c = crop_at(gone, box, pad)
        if c is not None:
            out.append((c, False, f"empty {tag}"))

    for hue in SUBTLE_HUES:
        imp = db._swap_instance(frame, box, hue)
        for pad in PADS:
            c = crop_at(imp, box, pad)
            if c is not None:
                out.append((c, False, f"impostor+{hue} pad{pad}"))

    for name, img in (("shadow", db._shadow(frame, 0.75, (0, 0))),
                      ("glare", db._glare(frame, 0.75)),
                      ("cool", db._white_balance(frame, 0.42))):
        for pad in PADS:
            c = crop_at(img, box, pad)
            if c is not None:
                out.append((c, True, f"target {name} pad{pad}"))
    return out


def main():
    import torch
    from run_study import make_fast_verifier

    frame = cv2.imread("_live_now.jpg")
    if frame is None:
        raise SystemExit("ABORT: cannot read _live_now.jpg")
    target = "red cup"
    found = db._target_box(frame, target)
    if not found:
        raise SystemExit(f"ABORT: {target!r} not in the base frame")
    box = found[1]

    crops = build(frame, box, target)
    pos = sum(1 for _, l, _ in crops if l)
    majority = max(pos, len(crops) - pos) / len(crops)
    print(f"{len(crops)} hard crops ({pos} yes / {len(crops) - pos} no)")
    print(f"majority-class baseline {majority:.3f} -- beat this or the "
          f"verifier is guessing\n")
    prompt = PROMPT.format(target)

    rows = {}
    for mp in BUDGETS:
        print(f"--- max_pixels={mp} ---")
        torch.cuda.empty_cache()
        v = make_fast_verifier(max_pixels=mp)
        try:
            for c, _, _ in crops[:WARMUP]:
                v(prompt, c)
            times, correct = [], 0
            per_tag = {}
            for c, lbl, tag in crops:
                t0 = time.time()
                ans = v(prompt, c)
                times.append(time.time() - t0)
                ok = str(ans).strip().lower().startswith("y") == lbl
                correct += ok
                bucket = tag.split()[0]
                b = per_tag.setdefault(bucket, [0, 0])
                b[0] += ok
                b[1] += 1
            mean = sum(times) / len(times)
            rows[str(mp)] = {"mean_s": round(mean, 3),
                             "acc": round(correct / len(crops), 3),
                             "correct": correct, "n": len(crops),
                             "per_bucket": {k: f"{a}/{b}"
                                            for k, (a, b) in per_tag.items()}}
            print(f"  {mean:.3f}s   acc {correct}/{len(crops)} "
                  f"({correct / len(crops):.3f})")
            for k, (a, b) in sorted(per_tag.items()):
                print(f"     {k:<12} {a}/{b}")
        finally:
            del v
            gc.collect()
            torch.cuda.empty_cache()
        json.dump(rows, open(OUT, "w"), indent=2)

    print("\n" + "=" * 62)
    print("DOES A HARDER SET SEPARATE THE BUDGETS?")
    print("=" * 62)
    lo, hi = rows.get(str(BUDGETS[0])), rows.get(str(BUDGETS[1]))
    if not (lo and hi):
        print("incomplete run")
        return
    print(f"{'max_pixels':>12}{'latency':>10}{'acc':>10}")
    for k in (str(BUDGETS[0]), str(BUDGETS[1])):
        print(f"{k:>12}{rows[k]['mean_s']:>10.3f}{rows[k]['acc']:>10.3f}")
    d = hi["correct"] - lo["correct"]
    saving = (hi["mean_s"] - lo["mean_s"]) / hi["mean_s"] * 100
    print()
    if d <= 0:
        print(f"  {BUDGETS[0]} matches or beats {BUDGETS[1]} on a set built to "
              f"punish low\n  resolution ({lo['correct']} vs {hi['correct']} "
              f"of {lo['n']}). ADOPT {BUDGETS[0]}: {saving:.0f}% faster, and "
              f"now\n  on evidence rather than on a saturated tie.")
    elif d <= 2:
        print(f"  {BUDGETS[1]} leads by {d}/{lo['n']} -- within noise at this "
              f"n. The {saving:.0f}%\n  saving is probably safe but is not "
              f"demonstrated. Keep {BUDGETS[1]} unless\n  latency becomes the "
              f"binding constraint.")
    else:
        print(f"  {BUDGETS[1]} leads by {d}/{lo['n']}. The saturation WAS "
              f"hiding a real\n  difference -- resolution matters once the "
              f"target is small or degraded.\n  KEEP {BUDGETS[1]}; the 21% was "
              f"an artifact of an easy test set.")
    if max(lo["acc"], hi["acc"]) <= majority + 0.02:
        print(f"\n  !! Neither beats the majority baseline ({majority:.3f}). "
              f"The set is too\n     hard to rank anything -- report that, do "
              f"not mine the buckets.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
