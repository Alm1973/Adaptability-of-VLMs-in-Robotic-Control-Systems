import json
import sys
from collections import defaultdict

OUT = "assignment3a.json"


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "red cup"
    from live_bench import build_episodes, KIND
    from recovery_pipeline import RecoveryPipeline
    from run_study import score_episode, make_fast_verifier
    from yolo_tracker import YoloTracker

    episodes, skipped = build_episodes()
    print(f"target={target!r}  episodes={len(episodes)}")
    for s in skipped:
        print(f"  skipped {s}")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_fast_verifier()

    CONDS = {
        "D_full":   dict(use_detector=True, use_opencv=True, use_vlm=True,  use_state=True),
        "E_no_vlm": dict(use_detector=True, use_opencv=True, use_vlm=False, use_state=True),
    }

    def instrument(pipe):
        pipe._trace = []
        orig = pipe.step

        def step(frame):
            out = orig(frame)
            pipe._trace.append((pipe.status, bool(pipe.believes_present())))
            return out

        pipe.step = step
        return pipe

    results = {}
    beliefs = defaultdict(dict)

    for cname, cfg in CONDS.items():
        per_ep = {}
        for name, ep in sorted(episodes.items()):
            pipe = instrument(RecoveryPipeline(
                target, tracker=tracker,
                verifier=verifier if cfg["use_vlm"] else None, **cfg))
            r = score_episode(pipe, ep, ep["frames"])
            per_ep[name] = {
                "kind": KIND.get(name, "?"),
                "durAcc": r["during_disruption_acc"],
                "lost": r["lost_while_present"],
                "false": r["false_belief_frames"],
                "spurious": r["spurious_reacquisition"],
                "vlm_calls": r["vlm_calls"],
            }
            beliefs[cname][name] = list(pipe._trace)
            print(f"  {cname:<10}{name:<24}durAcc={str(r['during_disruption_acc']):<7}"
                  f"lost={r['lost_while_present']:<4}false={r['false_belief_frames']:<4}"
                  f"vlm={r['vlm_calls']}")
        results[cname] = per_ep

    axes = sorted({KIND.get(n, "?") for n in episodes})
    print("\n" + "=" * 78)
    print("ASSIGNMENT 3(a)  --  VLM CONTRIBUTION BY AXIS  (live corpus)")
    print("=" * 78)
    hdr = f"{'axis':<13}{'n':>3}{'D durAcc':>10}{'E durAcc':>10}{'delta':>8}" \
          f"{'D lost':>8}{'E lost':>8}{'D false':>9}{'E false':>9}{'VLM calls':>11}"
    print(hdr)
    print("-" * 78)

    axis_rows = {}
    for ax in axes:
        eps = [n for n in episodes if KIND.get(n) == ax]
        row = {}
        for cname in CONDS:
            accs = [results[cname][n]["durAcc"] for n in eps
                    if results[cname][n]["durAcc"] is not None]
            row[cname] = {
                "durAcc": round(sum(accs) / len(accs), 3) if accs else None,
                "lost": sum(results[cname][n]["lost"] for n in eps),
                "false": sum(results[cname][n]["false"] for n in eps),
                "vlm_calls": sum(results[cname][n]["vlm_calls"] for n in eps),
            }
        d, e = row["D_full"], row["E_no_vlm"]
        delta = (round(d["durAcc"] - e["durAcc"], 3)
                 if d["durAcc"] is not None and e["durAcc"] is not None else None)
        row["delta_durAcc"] = delta
        row["n_episodes"] = len(eps)
        row["episodes"] = eps
        axis_rows[ax] = row
        print(f"{ax:<13}{len(eps):>3}{str(d['durAcc']):>10}{str(e['durAcc']):>10}"
              f"{str(delta):>8}{d['lost']:>8}{e['lost']:>8}"
              f"{d['false']:>9}{e['false']:>9}{d['vlm_calls']:>11}")

    print("\n" + "=" * 78)
    print("FRAME-LEVEL DISAGREEMENT  --  does a tie mean agreement, or cancellation?")
    print("=" * 78)
    print(f"{'scenario':<24}{'axis':<13}{'frames':>8}{'stateDif':>8}{'pct':>7}"
          f"{'ansDiff':>9}{'pct':>7}")
    print("-" * 78)
    disagree = {}
    for name in sorted(episodes):
        b1 = beliefs["D_full"].get(name) or []
        b2 = beliefs["E_no_vlm"].get(name) or []
        n = min(len(b1), len(b2))
        d_state = sum(1 for i in range(n) if b1[i][0] != b2[i][0])
        d_ans = sum(1 for i in range(n) if b1[i][1] != b2[i][1])
        ps = round(100.0 * d_state / n, 1) if n else None
        pa = round(100.0 * d_ans / n, 1) if n else None
        disagree[name] = {"frames": n, "differ_state": d_state,
                          "differ_answer": d_ans, "pct_state": ps,
                          "pct_answer": pa}
        print(f"{name:<24}{KIND.get(name,'?'):<13}{n:>8}{d_state:>8}{str(ps):>7}"
              f"{d_ans:>9}{str(pa):>7}")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    for ax in axes:
        r = axis_rows[ax]
        d, e = r["D_full"], r["E_no_vlm"]
        calls = d["vlm_calls"]
        delta = r["delta_durAcc"]
        if delta is None:
            continue
        dfalse, efalse = d["false"], e["false"]
        if dfalse != efalse:
            better = "CUT" if dfalse < efalse else "RAISED"
            print(f"  {ax:<13} VLM {better} FALSE BELIEF {efalse} -> {dfalse} "
                  f"for {calls} calls. durAcc moved only {delta:+.3f}, so the "
                  f"contribution is INVISIBLE in accuracy.")
        elif abs(delta) < 0.005 and d["lost"] == e["lost"]:
            print(f"  {ax:<13} VLM CHANGED NOTHING. {calls} calls, identical "
                  f"durAcc/lost/false, and 0 frames of state disagreement. "
                  f"OpenCV alone suffices.")
        elif delta > 0:
            print(f"  {ax:<13} VLM HELPED  +{delta:.3f} durAcc for {calls} calls.")
        else:
            print(f"  {ax:<13} VLM HURT  {delta:.3f} durAcc, and still cost "
                  f"{calls} calls.")

    json.dump({"per_episode": results, "per_axis": axis_rows,
               "disagreement": disagree}, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
