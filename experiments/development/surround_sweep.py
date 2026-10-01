import json

import disruption_bench as db
from run_study import score_episode, make_verifier

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
OUT = "surround_sweep.json"


def main():
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    idx = db.load()
    targets = {e["meta"].get("target", "unknown") for e in idx.values()}
    if len(targets) != 1 or "unknown" in targets:
        raise SystemExit(f"ABORT: inconsistent target {targets}")
    target = targets.pop()
    print(f"{len(idx)} episodes, target={target!r}")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_verifier()

    rows = {}
    for th in THRESHOLDS:
        pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                                use_detector=True, use_opencv=True,
                                use_vlm=True, use_state=True,
                                surround_similar=th)
        lost = false = 0
        das = []
        occ_lost = occ_false = 0
        for name, ep in idx.items():
            r = score_episode(pipe, ep, ep["frames"])
            lost += r["lost_while_present"]
            false += r["false_belief_frames"]
            if r["during_disruption_acc"] is not None:
                das.append(r["during_disruption_acc"])
            if ep["meta"]["kind"] in ("occlusion", "identity"):
                occ_lost += r["lost_while_present"]
                occ_false += r["false_belief_frames"]
        acc = round(sum(das) / len(das), 3) if das else None
        rows[str(th)] = {"lost": lost, "false": false, "total": lost + false,
                         "occ_lost": occ_lost, "occ_false": occ_false,
                         "occ_total": occ_lost + occ_false, "during_acc": acc}
        print(f"  thr={th:<5} lost={lost:<4} false={false:<4} "
              f"total={lost + false:<4} | occlusion-only "
              f"lost={occ_lost:<4} false={occ_false:<4} durAcc={acc}")
        json.dump(rows, open(OUT, "w"), indent=2)

    print("\n" + "=" * 66)
    print("SURROUND_SIMILAR TRADEOFF")
    print("  lost  = abandons a target that is merely covered")
    print("  false = keeps asserting a target that was removed")
    print("=" * 66)
    print(f"{'thr':>6}{'lost':>7}{'false':>7}{'total':>7}"
          f"{'occLost':>10}{'occFalse':>10}{'durAcc':>9}")
    print("-" * 56)
    best = best_occ = None
    for k, v in rows.items():
        print(f"{k:>6}{v['lost']:>7}{v['false']:>7}{v['total']:>7}"
              f"{v['occ_lost']:>10}{v['occ_false']:>10}"
              f"{str(v['during_acc']):>9}")
        if best is None or v["total"] < rows[best]["total"]:
            best = k
        if best_occ is None or v["occ_total"] < rows[best_occ]["occ_total"]:
            best_occ = k

    flat = len({v["total"] for v in rows.values()}) == 1
    print(f"\nlowest total error at thr={best}; on occlusion+identity "
          f"episodes only, thr={best_occ}")
    if flat:
        print("WARNING: the curve is FLAT -- every threshold scores the same. "
              "That\nmeans this branch is not reached on this corpus, and the "
              "shipped value\nis unjustified rather than validated. Do not "
              "report it as swept.")
    else:
        span = max(v["total"] for v in rows.values()) - \
            min(v["total"] for v in rows.values())
        print(f"error spans {span} frames across the range, so the parameter "
              f"does do work.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
