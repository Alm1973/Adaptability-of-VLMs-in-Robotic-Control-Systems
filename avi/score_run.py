import argparse
import json
import os
import sys
from collections import defaultdict

OUT_NAME = "ablation_scores.json"


def build_episodes(run_dir):
    rp = os.path.join(run_dir, "results.json")
    if not os.path.exists(rp):
        raise SystemExit(f"ABORT: no {rp}")
    rows = json.load(open(rp))

    cfgp = os.path.join(run_dir, "experiment_config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        if cfg.get("smoke"):
            raise SystemExit(
                f"ABORT: {run_dir} is a SMOKE run (short phases, 1 rep). It "
                f"is a harness test, not data. Score a real run instead.")

    episodes = {}
    skipped = []
    for row in rows:
        if row.get("verdict") in ("INVALID", "ABORTED"):
            skipped.append(f"trial {row['trial_no']} ({row['scenario']}) "
                           f"-- {row['verdict']}")
            continue
        frames = row.get("frames") or []
        if not frames:
            skipped.append(f"trial {row['trial_no']} -- no frames")
            continue

        paths, gt = [], []
        disrupt_start = recover_start = None
        for i, f in enumerate(frames):
            p = os.path.join(run_dir, f["raw"])
            if not os.path.exists(p):
                skipped.append(f"trial {row['trial_no']} -- missing {f['raw']}")
                paths = []
                break
            paths.append(p)
            gt.append({"present": None if f["edge"] else bool(f["present"])})
            if f["phase"] == "disrupt" and disrupt_start is None:
                disrupt_start = i
            if f["phase"] == "recover" and recover_start is None:
                recover_start = i
        if not paths:
            continue
        if disrupt_start is None:
            disrupt_start = 0
        if recover_start is None:
            recover_start = len(paths)

        key = f"{row['trial_no']:03d}_{row['scenario']}_rep{row['rep']}"
        episodes[key] = {
            "frames": paths, "gt": gt,
            "meta": {"disrupt_start": disrupt_start,
                     "recover_start": recover_start},
            "scenario": row["scenario"], "rep": row["rep"],
            "clutter": row.get("clutter"),
        }
    return episodes, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--conditions", default="D_full,E_no_vlm",
                    help="comma-separated names from run_study.CONDITIONS")
    ap.add_argument("--decay", default="",
                    help="also sweep decay_frames, e.g. 12,30,60. Applies to "
                         "D_full only.")
    ap.add_argument("--target", default="red cup")
    args = ap.parse_args()

    from recovery_pipeline import RecoveryPipeline
    from run_study import CONDITIONS, score_episode, make_fast_verifier
    from yolo_tracker import YoloTracker

    try:
        from final_experiment import AXIS
    except Exception:
        AXIS = {}

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad}; "
                         f"choose from {list(CONDITIONS)}")
    decays = [int(d) for d in args.decay.split(",") if d.strip()]

    episodes, skipped = build_episodes(args.run_dir)
    if not episodes:
        raise SystemExit("ABORT: no usable episodes in that run")
    total_frames = sum(len(e["frames"]) for e in episodes.values())
    scored_frames = sum(1 for e in episodes.values()
                        for g in e["gt"] if g["present"] is not None)
    absent = sum(1 for e in episodes.values()
                 for g in e["gt"] if g["present"] is False)
    print(f"{len(episodes)} episodes, {total_frames} frames, "
          f"{scored_frames} scored ({total_frames - scored_frames} edge)")
    print(f"absent frames: {absent} "
          f"({100.0 * absent / scored_frames:.1f}% of scored)")
    if absent / max(scored_frames, 1) < 0.15:
        print("  ** WARNING: few absent frames. False belief -- the primary")
        print("     metric -- is barely measurable on this run. **")
    for s in skipped:
        print(f"  skipped {s}")

    tracker = YoloTracker(default_target=args.target)
    tracker.set_targets([args.target], allow_unreliable=True, quiet=True)
    print("\nloading verifier...")
    verifier = make_fast_verifier()
    gen = getattr(verifier, "generate", None)

    variants = [(c, CONDITIONS[c], None) for c in conds]
    for d in decays:
        variants.append((f"D_full(decay={d})", CONDITIONS["D_full"], d))

    results = {}
    for label, cfg, decay in variants:
        print(f"\n--- {label} ---")
        per_ep = {}
        for key, ep in sorted(episodes.items()):
            kw = dict(cfg)
            use_probe = kw.pop("use_region_probe", False)
            if decay is not None:
                kw["decay_frames"] = decay
            pipe = RecoveryPipeline(
                args.target, tracker=tracker,
                verifier=verifier if kw.get("use_vlm") else None,
                use_region_probe=use_probe,
                probe_verifier=gen if use_probe else None, **kw)
            r = score_episode(pipe, ep, ep["frames"])
            per_ep[key] = {"scenario": ep["scenario"], "rep": ep["rep"],
                           "axis": AXIS.get(ep["scenario"], "?"),
                           "durAcc": r["during_disruption_acc"],
                           "false": r["false_belief_frames"],
                           "lost": r["lost_while_present"],
                           "spurious": r["spurious_reacquisition"],
                           "vlm_calls": r["vlm_calls"]}
            print(f"  {key:<32} acc={str(r['during_disruption_acc']):<7}"
                  f"false={r['false_belief_frames']:<4}"
                  f"lost={r['lost_while_present']:<4}"
                  f"vlm={r['vlm_calls']}")
        results[label] = per_ep

    def rollup(per_ep, keyfn):
        agg = defaultdict(lambda: {"n": 0, "acc": [], "false": 0, "lost": 0,
                                   "vlm": 0, "spurious": 0})
        for v in per_ep.values():
            a = agg[keyfn(v)]
            a["n"] += 1
            if v["durAcc"] is not None:
                a["acc"].append(v["durAcc"])
            a["false"] += v["false"]
            a["lost"] += v["lost"]
            a["vlm"] += v["vlm_calls"]
            a["spurious"] += bool(v["spurious"])
        for a in agg.values():
            a["mean_acc"] = (round(sum(a["acc"]) / len(a["acc"]), 3)
                             if a["acc"] else None)
            del a["acc"]
        return dict(agg)

    print("\n" + "=" * 78)
    print("BY SCENARIO   (false belief is the PRIMARY metric)")
    print("=" * 78)
    scen = sorted({v["scenario"] for v in next(iter(results.values())).values()})
    head = f"{'scenario':<17}" + "".join(f"{l[:16]:>19}" for l in results)
    print(head)
    print("-" * len(head))
    for s in scen:
        line = f"{s:<17}"
        for label, per_ep in results.items():
            r = rollup(per_ep, lambda v: v["scenario"]).get(s)
            line += (f"{str(r['mean_acc']):>8}/{r['false']:<10}" if r
                     else f"{'-':>19}")
        print(line)
    print("\n(cells are  mean_durAcc / total_false_belief_frames)")

    print("\n" + "=" * 78)
    print("BY AXIS  -- the three families the research question names")
    print("=" * 78)
    axes = sorted({v["axis"] for v in next(iter(results.values())).values()})
    head = f"{'axis':<15}" + "".join(f"{l[:16]:>19}" for l in results)
    print(head)
    print("-" * len(head))
    for a in axes:
        line = f"{a:<15}"
        for label, per_ep in results.items():
            r = rollup(per_ep, lambda v: v["axis"]).get(a)
            line += (f"{str(r['mean_acc']):>8}/{r['false']:<10}" if r
                     else f"{'-':>19}")
        print(line)

    if "D_full" in results and "E_no_vlm" in results:
        print("\n" + "=" * 78)
        print("WHAT THE VLM CONTRIBUTED  (D_full minus E_no_vlm, per axis)")
        print("=" * 78)
        d = rollup(results["D_full"], lambda v: v["axis"])
        e = rollup(results["E_no_vlm"], lambda v: v["axis"])
        print(f"{'axis':<15}{'n':>4}{'D acc':>8}{'E acc':>8}{'d acc':>8}"
              f"{'D false':>9}{'E false':>9}{'calls':>7}")
        print("-" * 78)
        for a in axes:
            if a not in d or a not in e:
                continue
            da, ea = d[a]["mean_acc"], e[a]["mean_acc"]
            delta = (round(da - ea, 3) if da is not None and ea is not None
                     else None)
            print(f"{a:<15}{d[a]['n']:>4}{str(da):>8}{str(ea):>8}"
                  f"{str(delta):>8}{d[a]['false']:>9}{e[a]['false']:>9}"
                  f"{d[a]['vlm']:>7}")
        print("\nRead the FALSE BELIEF columns first. On the pilot corpus the")
        print("VLM's entire contribution was there and invisible in accuracy:")
        print("removing it RAISED accuracy while multiplying false belief 4.6x.")

    out = os.path.join(args.run_dir, OUT_NAME)
    json.dump({"conditions": list(results), "per_episode": results,
               "skipped": skipped,
               "corpus": {"episodes": len(episodes), "frames": total_frames,
                          "scored": scored_frames, "absent": absent}},
              open(out, "w"), indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
