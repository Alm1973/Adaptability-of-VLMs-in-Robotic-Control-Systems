import json

import cv2

import disruption_bench as db
from run_study import score_episode, make_verifier

DECAYS = [0, 2, 4, 6, 8, 12, 18, 26]
OUT = "decay_sweep.json"


def main():
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    idx = db.load()
    targets = {e["meta"].get("target", "unknown") for e in idx.values()}
    target = targets.pop()
    print(f"{len(idx)} episodes, target={target!r}")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    verifier = make_verifier()

    rows = {}
    for d in DECAYS:
        pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                                use_detector=True, use_opencv=True,
                                use_vlm=True, use_state=True, decay_frames=d)
        lost = false = 0
        das = []
        per = {}
        strata = {}
        for name, ep in idx.items():
            r = score_episode(pipe, ep, ep["frames"])
            lost += r["lost_while_present"]
            false += r["false_belief_frames"]
            if r["during_disruption_acc"] is not None:
                das.append(r["during_disruption_acc"])
            per[name] = {"lost": r["lost_while_present"],
                         "false": r["false_belief_frames"],
                         "final": r["final_status"]}
            s = strata.setdefault(ep["meta"]["disrupt_frames"],
                                  {"lost": 0, "false": 0, "n": 0})
            s["lost"] += r["lost_while_present"]
            s["false"] += r["false_belief_frames"]
            s["n"] += 1
        for s in strata.values():
            s["total"] = s["lost"] + s["false"]
        acc = round(sum(das) / len(das), 3) if das else None
        rows[str(d)] = {"lost": lost, "false": false, "total": lost + false,
                        "during_acc": acc, "strata": strata,
                        "per_episode": per}
        print(f"  decay={d:<3} lost={lost:<4} false={false:<4} "
              f"total={lost + false:<4} durAcc={acc}   "
              + "  ".join(f"d{k}:{v['total']}"
                          for k, v in sorted(strata.items())))
        json.dump(rows, open(OUT, "w"), indent=2)

    print("\n" + "=" * 60)
    print("DECAY TRADEOFF   (lost = abandons a hidden target,")
    print("                  false = acts on a removed one)")
    print("=" * 60)
    print(f"{'decay':>6}{'lost':>7}{'false':>7}{'total':>7}{'durAcc':>9}")
    print("-" * 36)
    best = None
    for k, v in rows.items():
        print(f"{k:>6}{v['lost']:>7}{v['false']:>7}{v['total']:>7}"
              f"{str(v['during_acc']):>9}")
        if best is None or v["total"] < rows[best]["total"]:
            best = k
    print(f"\nlowest total error at decay={best} "
          f"(lost={rows[best]['lost']}, false={rows[best]['false']})")

    lengths = sorted({int(k) for r in rows.values() for k in r["strata"]})
    print("\n" + "=" * 60)
    print("CONFOUND TEST -- does the optimum just track disruption length?")
    print("=" * 60)
    hdr = f"{'decay':>6}" + "".join(f"{'dlen=' + str(L):>12}" for L in lengths)
    print(hdr); print("-" * len(hdr))
    for k, v in rows.items():
        cells = "".join(f"{v['strata'].get(L, {}).get('total', 0):>12}"
                        for L in lengths)
        print(f"{k:>6}{cells}")

    per_stratum_best = {}
    for L in lengths:
        b = min(rows, key=lambda k: rows[k]["strata"].get(L, {})
                .get("total", 10 ** 9))
        per_stratum_best[L] = int(b)
    print("\nbest decay per stratum: " +
          ", ".join(f"dlen={L} -> decay={b}"
                    for L, b in per_stratum_best.items()))

    vals = list(per_stratum_best.values())
    tracks = all(abs(per_stratum_best[L] - L) <= 2 for L in lengths)
    if tracks:
        print("\nVERDICT: the optimum TRACKS the disruption length. This is an "
              "ARTIFACT.\nNo single decay value generalises across disruption "
              "durations, so a fixed\ntimeout is the wrong shape of solution -- "
              "the honest report is that the\nparameter must be adaptive, not "
              "that it equals any number here.")
    elif max(vals) - min(vals) <= 4:
        print(f"\nVERDICT: the per-stratum optima CLUSTER ({vals}) rather than "
              "tracking\nlength. That makes decay a real property of the "
              "occluded/removed decision\nand this value defensible as derived "
              "-- which is what decision 6 asked for.")
    else:
        print(f"\nVERDICT: optima are scattered ({vals}) without tracking "
              "length. Neither\nartifact nor a clean constant; report the "
              "curve, not a single value.")
    print(f"\nn={sum(next(iter(rows.values()))['strata'][L]['n'] for L in lengths)}"
          f" episodes across {len(lengths)} disruption lengths. This "
          f"identifies the\nSHAPE of the tradeoff; it is still one scene and "
          f"one target.")


if __name__ == "__main__":
    main()
