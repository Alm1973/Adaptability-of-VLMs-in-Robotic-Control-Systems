import json
import os
import random
import sys
from collections import Counter

import cv2

OUT = "search_prompt_sweep.json"
FRAME_DIR = "live_disruption_frames"
N = 40
SEED = 20260828
SIDE_FRAC = 0.55

PROMPTS = {
    "current": (
        "A camera is looking for a {t}, which is NOT in view. Based on what "
        "you can see, which way should the camera turn to find it? Already "
        "tried: none. Answer with exactly one word: left, right, up, or down."),
    "no_options_list": (
        "A camera is looking for a {t}. It is not in this view. Which "
        "direction should the camera turn? Answer left or right."),
    "describe_then_choose": (
        "This is the {side} portion of a wider scene. A {t} is just outside "
        "this view. Is it off the left edge or the right edge? Answer left or "
        "right."),
    "edge_cue": (
        "Look at this image. A {t} is just outside the frame, immediately "
        "beyond one of the vertical edges. Which edge is it beyond -- the "
        "left edge or the right edge? Answer with one word."),
    "spatial_continuation": (
        "The desk in this photo continues past one side of the frame. A {t} "
        "sits on the part of the desk you cannot see. Which side does the "
        "desk continue toward? Answer left or right."),
    "forced_choice_swapped": (
        "A camera is looking for a {t}, which is NOT in view. Which way "
        "should the camera turn to find it? Answer with exactly one word: "
        "right or left."),
}


def build_set(tracker, target, rng, n):
    files = [f for f in sorted(os.listdir(FRAME_DIR)) if f.endswith(".jpg")]
    rng.shuffle(files)
    want = n // 2
    out, got = [], Counter()
    for fn in files:
        if got["left"] >= want and got["right"] >= want:
            break
        frame = cv2.imread(os.path.join(FRAME_DIR, fn))
        if frame is None:
            continue
        det = tracker.find(frame, target)
        if det is None:
            continue
        H, W = frame.shape[:2]
        bx, by, bw, bh = det["box"]
        cw = int(W * SIDE_FRAC)
        if bx >= cw + 5 and got["right"] < want:
            out.append((frame[:, :cw].copy(), "right", "left"))
            got["right"] += 1
            continue
        if bx + bw <= W - cw - 5 and got["left"] < want:
            out.append((frame[:, W - cw:].copy(), "left", "right"))
            got["left"] += 1
    rng.shuffle(out)
    return out


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "red cup"
    from run_study import make_fast_verifier
    from yolo_tracker import YoloTracker

    rng = random.Random(SEED)
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    gen = make_fast_verifier().generate

    trials = build_set(tracker, target, rng, N)
    truths = Counter(t for _, t, _ in trials)
    print(f"\ntest set: {len(trials)} crops  truth balance={dict(truths)}")
    if len(trials) < 8 or len(truths) < 2:
        raise SystemExit("ABORT: could not build a balanced set; "
                         "not reporting a number from an unbalanced one")

    results = {}
    print(f"\n{'prompt':<26}{'n':>4}{'acc':>7}{'chance':>8}   answer histogram")
    print("-" * 78)
    for pname, tmpl in PROMPTS.items():
        correct, ans = 0, Counter()
        for crop, truth, side in trials:
            q = tmpl.format(t=target, side=side)
            raw = gen(q, crop)
            low = (raw or "").strip().lower()
            got = None
            for d in ("left", "right"):
                if d in low:
                    got = d
                    break
            ans[got or "<none>"] += 1
            correct += (got == truth)
        acc = round(correct / len(trials), 3)
        results[pname] = {"n": len(trials), "correct": correct, "accuracy": acc,
                          "chance": 0.5, "answers": dict(ans),
                          "prompt": tmpl}
        print(f"{pname:<26}{len(trials):>4}{acc:>7}{0.5:>8}   {dict(ans)}")

    best = max(results.items(), key=lambda kv: kv[1]["accuracy"])
    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print(f"  best prompt: {best[0]}  acc={best[1]['accuracy']} (chance 0.50)")
    if best[1]["accuracy"] <= 0.60:
        print("  -> NO PROMPT RECOVERS THE SIGNAL. The constant answer is not")
        print("     an artifact of how the question was worded. This model")
        print("     cannot do off-frame direction on this scene, so directional")
        print("     search cannot be built on it as-is. Fixing the wiring alone")
        print("     would connect a mechanism that still does not work.")
    else:
        print("  -> A PROMPT RECOVERS SIGNAL. The original wording was the")
        print("     problem. Adopt the better prompt, fix the wiring, then the")
        print("     live cluttered-search test is worth its session.")

    json.dump({"results": results, "truth_balance": dict(truths),
               "params": {"N": N, "SEED": SEED, "SIDE_FRAC": SIDE_FRAC}},
              open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
