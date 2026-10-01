import json
import os
import random
import sys
from collections import Counter, defaultdict

import cv2

OUT = "assignment3b_offline.json"
FRAME_DIR = "live_disruption_frames"

CLUTTERED_PREFIXES = ("clutter_similar", "distractor")
CLEAN_PREFIXES = ("occlusion_full", "occlusion_remove", "occlude_book",
                  "occlude_paper", "occlude_box", "occlude_jacket",
                  "lighting", "camera_pose")

N_PER_GROUP = 40
SEED = 20260828
CONSTANCY_LIMIT = 0.90


def gather(prefixes, rng, n):
    out = [os.path.join(FRAME_DIR, f) for f in sorted(os.listdir(FRAME_DIR))
           if f.endswith(".jpg") and f.startswith(prefixes)]
    rng.shuffle(out)
    return out[:n]


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "red cup"
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier
    from yolo_tracker import YoloTracker

    rng = random.Random(SEED)
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    fast = make_fast_verifier()
    gen = getattr(fast, "generate", None)
    if gen is None:
        raise SystemExit("ABORT: fast verifier exposes no .generate")

    groups = {"clean": gather(CLEAN_PREFIXES, rng, N_PER_GROUP),
              "cluttered": gather(CLUTTERED_PREFIXES, rng, N_PER_GROUP)}
    report = {}

    print("\n" + "=" * 74)
    print("PART 1  --  what search_direction() returns in the SHIPPING config")
    print("=" * 74)
    shipping = RecoveryPipeline(target, tracker=tracker, verifier=fast,
                                use_detector=True, use_opencv=True,
                                use_vlm=True, use_state=True)
    ship_ans, raw_ans = Counter(), Counter()
    probe = groups["clean"][:12] + groups["cluttered"][:12]
    for path in probe:
        img = cv2.imread(path)
        if img is None:
            continue
        ship_ans[repr(shipping.search_direction(img))] += 1
        raw_ans[repr(fast("A camera is looking for a red cup, which is NOT in "
                          "view. Which way should the camera turn to find it? "
                          "Answer with exactly one word: left, right, up, or "
                          "down.", img))] += 1
    print(f"  frames probed: {sum(ship_ans.values())}")
    print(f"  raw verifier answers      : {dict(raw_ans)}")
    print(f"  search_direction() returns: {dict(ship_ans)}")
    dead = set(ship_ans) <= {"None"}
    print(f"\n  -> {'DEAD: returns None on every frame.' if dead else 'returns real directions.'}")
    if dead:
        print("     The VLM has never steered a search in any recorded run.")
    report["part1_wiring"] = {"n": sum(ship_ans.values()),
                              "raw_verifier_answers": dict(raw_ans),
                              "search_direction_returns": dict(ship_ans),
                              "dead": bool(dead)}

    print("\n" + "=" * 74)
    print("PART 2  --  wired to .generate, does the answer depend on the image?")
    print("=" * 74)
    fixed = RecoveryPipeline(target, tracker=tracker, verifier=gen,
                             use_detector=True, use_opencv=True,
                             use_vlm=True, use_state=True)
    part2, all_ans = {}, Counter()
    for gname, paths in groups.items():
        ans = Counter()
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            ans[fixed.search_direction(img) or "<none>"] += 1
        n = sum(ans.values())
        top, topn = ans.most_common(1)[0]
        part2[gname] = {"n": n, "answers": dict(ans),
                        "modal_answer": top,
                        "modal_share": round(topn / n, 3) if n else None}
        all_ans.update(ans)
        print(f"  {gname:<11} n={n:<4} {dict(ans)}")
        print(f"              modal '{top}' = {round(100*topn/n,1)}% of answers")
    n_all = sum(all_ans.values())
    top_all, topn_all = all_ans.most_common(1)[0]
    share = topn_all / n_all if n_all else 0
    constant = share >= CONSTANCY_LIMIT
    print(f"\n  overall: {dict(all_ans)}")
    print(f"  -> modal answer '{top_all}' on {round(100*share,1)}% of "
          f"{n_all} real frames")
    if constant:
        print("     CONSTANT FUNCTION. The direction does not depend on the")
        print("     image, so there is no scene reasoning to amplify.")
    report["part2_constancy"] = {"per_group": part2, "overall": dict(all_ans),
                                 "modal_answer": top_all,
                                 "modal_share": round(share, 3),
                                 "is_constant": bool(constant)}

    print("\n  sanity check -- same frames, open description:")
    for path in probe[:4]:
        img = cv2.imread(path)
        cap = gen("Describe what you see in one short sentence.", img)
        print(f"    {os.path.basename(path)[:30]:<32} -> {str(cap)[:60]}")

    print("\n" + "=" * 74)
    print("PART 3  --  geometric accuracy test")
    print("=" * 74)
    if constant:
        print("  SKIPPED. Part 2 says the answer is constant, so an accuracy")
        print("  number here would only measure which constant it happens to")
        print("  be against whatever ground truth the crops happen to produce.")
        print("  That is the exact artifact the first version of this script")
        print("  fell into (120/120 crops had truth 'down'). Nothing to add")
        print("  until the mechanism varies with the image.")
        report["part3"] = {"run": False, "reason": "constant answer in part 2"}
    else:
        print("  Answers vary -- geometry test is meaningful. Not implemented")
        print("  in this pass; rebuild with balanced per-direction sampling.")
        report["part3"] = {"run": False, "reason": "answers vary; harness TODO"}

    print("\n" + "=" * 74)
    print("VERDICT  --  Assignment 3(b)")
    print("=" * 74)
    if dead and constant:
        print("  There is NO directional-search advantage to amplify.")
        print("   1. As shipped, search_direction() returns None on every")
        print("      frame -- it asks a 4-way question of a yes/no comparator.")
        print("   2. Wired correctly it answers "
              f"'{top_all}' on {round(100*share,1)}% of frames regardless of "
              "content.")
        print("  The live cluttered-search experiment should NOT be run until")
        print("  both are fixed. It would be measuring a disconnected mechanism.")
    elif dead:
        print("  Mechanism disconnected as shipped, but varies once wired.")
        print("  Fix the wiring, then the live test is worth running.")
    else:
        print("  Mechanism live and varying. Proceed to the live test.")

    json.dump(report, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
