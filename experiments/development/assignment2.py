import json
import sys

import cv2

OUT = "assignment2.json"
OCC = ["occlusion_full", "occlusion_remove", "occlude_book", "occlude_paper",
       "occlude_box", "occlude_jacket", "identity_same_class"]
HAND_WORDS = ("hand", "finger", "palm", "wrist", "arm")
OBSERVED_WORDS = ("document", "notebook", "socks", "bookend", "lamp", "suit")


def score_set(pipe_factory, episodes, label):
    from run_study import score_episode
    rows, acc, lost, false = {}, [], 0, 0
    for name in OCC:
        ep = episodes.get(name)
        if ep is None:
            continue
        pipe = pipe_factory()
        r = score_episode(pipe, ep, ep["frames"])
        rows[name] = {"durAcc": r["during_disruption_acc"],
                      "lost": r["lost_while_present"],
                      "false": r["false_belief_frames"]}
        if r["during_disruption_acc"] is not None:
            acc.append(r["during_disruption_acc"])
        lost += r["lost_while_present"]
        false += r["false_belief_frames"]
    agg = round(sum(acc) / len(acc), 3) if acc else None
    print(f"  {label:<26} agg={str(agg):<7} lost={lost:<5} false={false}")
    return {"label": label, "aggregate": agg, "lost": lost, "false": false,
            "per_episode": rows}


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "red cup"
    from live_bench import build_episodes
    from recovery_pipeline import RecoveryPipeline
    from run_study import make_fast_verifier
    from yolo_tracker import YoloTracker

    episodes, skipped = build_episodes()
    episodes = {k: v for k, v in episodes.items() if k in OCC}
    print(f"{len(episodes)} occlusion episodes, target={target!r}")
    for s in skipped:
        print(f"  skipped {s}")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_fast_verifier()
    gen = getattr(verifier, "generate", None)

    base_words = RecoveryPipeline.OCCLUDER_WORDS
    print(f"baseline vocabulary ({len(base_words)}): {base_words}\n")

    def make(use_probe=False, decay=None, words=None, oracle=None):
        def factory():
            kw = dict(use_detector=True, use_opencv=True, use_vlm=True,
                      use_state=True)
            if decay is not None:
                kw["decay_frames"] = decay
            p = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                                 use_region_probe=use_probe,
                                 probe_verifier=(oracle or gen) if use_probe
                                 else None, **kw)
            if words is not None:
                p.OCCLUDER_WORDS = words
            return p
        return factory

    results = {}

    print("REFERENCE")
    results["D_baseline"] = score_set(make(False, 12), episodes,
                                      "D_full (timer=12)")
    results["F_baseline"] = score_set(make(True), episodes,
                                      "F_probe (current vocab)")

    print("\nT1  BREAK THE HAND MATCH  (vocab minus hand/finger/palm/wrist/arm)")
    nohand = tuple(w for w in base_words if w not in HAND_WORDS)
    print(f"  vocabulary now ({len(nohand)}): {nohand}")
    results["F_nohand"] = score_set(make(True, words=nohand), episodes,
                                    "F_probe (no hand words)")

    print("\nT2  TIMER SWEEP  (D_full, no probe, decay_frames varied)")
    for d in (12, 20, 30, 45, 60, 9999):
        lbl = f"D_full (timer={d})" if d < 9999 else "D_full (timer=INFINITE)"
        results[f"D_decay{d}"] = score_set(make(False, d), episodes, lbl)

    print("\nT3  ORACLE VOCABULARY  (add the words the VLM actually said)")
    rich = tuple(base_words) + OBSERVED_WORDS
    print(f"  added: {OBSERVED_WORDS}")
    results["F_oracle_vocab"] = score_set(make(True, words=rich), episodes,
                                          "F_probe (oracle vocab)")

    print("\nT4  ORACLE PROBE  (perfect hidden-vs-gone answers = the ceiling)")

    class OracleProbe:
        def __init__(self):
            self.answer = "hand"

        def __call__(self, prompt, crop):
            return self.answer

    oracle = OracleProbe()

    def oracle_factory_for(name):
        oracle.answer = "hand"
        return make(True, oracle=oracle)()

    rows, acc, lost, false = {}, [], 0, 0
    from run_study import score_episode
    for name in OCC:
        ep = episodes.get(name)
        if ep is None:
            continue
        p = oracle_factory_for(name)
        r = score_episode(p, ep, ep["frames"])
        rows[name] = {"durAcc": r["during_disruption_acc"],
                      "lost": r["lost_while_present"],
                      "false": r["false_belief_frames"]}
        if r["during_disruption_acc"] is not None:
            acc.append(r["during_disruption_acc"])
        lost += r["lost_while_present"]
        false += r["false_belief_frames"]
    agg = round(sum(acc) / len(acc), 3) if acc else None
    print(f"  {'F_probe (ORACLE always-occluder)':<26} agg={str(agg):<7} "
          f"lost={lost:<5} false={false}")
    results["F_oracle_probe"] = {"label": "F_probe (oracle always-occluder)",
                                 "aggregate": agg, "lost": lost,
                                 "false": false, "per_episode": rows}

    print("\n" + "=" * 76)
    print("HAND EPISODES ONLY  -- this is where the fork is decided")
    print("=" * 76)
    print(f"{'variant':<34}{'occlusion_full':>16}{'occlusion_remove':>18}")
    print("-" * 76)
    for k in ("D_baseline", "F_baseline", "F_nohand", "D_decay30",
              "D_decay60", "D_decay9999", "F_oracle_vocab"):
        r = results.get(k)
        if not r:
            continue
        a = r["per_episode"].get("occlusion_full", {}).get("durAcc")
        b = r["per_episode"].get("occlusion_remove", {}).get("durAcc")
        print(f"{r['label']:<34}{str(a):>16}{str(b):>18}")

    fb = results["F_baseline"]["per_episode"].get("occlusion_full", {})
    fn = results["F_nohand"]["per_episode"].get("occlusion_full", {})
    db = results["D_baseline"]["per_episode"].get("occlusion_full", {})
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    if fb.get("durAcc") is not None and fn.get("durAcc") is not None:
        drop = fb["durAcc"] - fn["durAcc"]
        print(f"  F with hand words   : {fb['durAcc']}")
        print(f"  F without hand words: {fn['durAcc']}")
        print(f"  D (timer=12)        : {db.get('durAcc')}")
        print(f"  drop when the match is broken: {drop:+.3f}")
        if fn["durAcc"] is not None and db.get("durAcc") is not None:
            if abs(fn["durAcc"] - db["durAcc"]) < 0.15:
                print("\n  -> STRING-MATCH ARTIFACT. Removing one word collapses F to D.")
                print("     F was not reasoning about occlusion; it was matching a token.")
                print("     Paper story: 'a fixable NLP bug', NOT 'sim does not transfer'.")
            elif drop < 0.15:
                print("\n  -> ARCHITECTURAL. Breaking the match barely moved it, so the")
                print("     advantage does not rest on that token.")
            else:
                print("\n  -> MIXED. Real drop, but not all the way to D. Both effects present;")
                print("     report the size of each rather than picking a side.")

    json.dump(results, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
