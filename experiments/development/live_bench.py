import json
import os
import re
import sys
from collections import Counter

import cv2

FRAMES = "live_disruption_frames"
GT_FILE = "live_disruption.json"
OUT = "live_bench.json"
PHASE_ORDER = ("baseline", "disrupt", "recover")

BAD_GT = {"impostor_colour": "recover phase never swaps the cup -- same red "
                             "cup visible throughout, GT absent is wrong"}


def build_episodes():
    if not os.path.isdir(FRAMES):
        raise SystemExit(f"ABORT: no {FRAMES}/")
    if not os.path.exists(GT_FILE):
        raise SystemExit(f"ABORT: no {GT_FILE} -- GT comes from the capture "
                         f"protocol, and cannot be reconstructed from frames.")
    gt_all = json.load(open(GT_FILE))
    if "records" in gt_all and isinstance(gt_all.get("records"), dict):
        gt_all = gt_all["records"]

    frames_by = {}
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(r"^(.*)_(baseline|disrupt|recover)_(\d+)\.jpg$", fn)
        if m:
            frames_by.setdefault((m.group(1), m.group(2)), []).append(
                (int(m.group(3)), os.path.join(FRAMES, fn)))
    for k in frames_by:
        frames_by[k] = [p for _, p in sorted(frames_by[k])]

    episodes, skipped = {}, []
    for scenario, recs in gt_all.items():
        if scenario in BAD_GT:
            skipped.append(f"{scenario} -- GT REJECTED: {BAD_GT[scenario]}")
            continue
        by_phase = {}
        for r in recs:
            by_phase.setdefault(r["phase"], []).append(r)

        missing = [p for p in PHASE_ORDER if not frames_by.get((scenario, p))]
        if missing:
            skipped.append(f"{scenario} (no frames for {'/'.join(missing)})")
            continue

        bad = [p for p in PHASE_ORDER
               if len(by_phase.get(p, [])) != len(frames_by[(scenario, p)])]
        if bad:
            skipped.append(f"{scenario} (GT/frame count mismatch in "
                           f"{'/'.join(bad)} -- alignment unsafe)")
            continue

        paths, gt, meta = [], [], {}
        for phase in PHASE_ORDER:
            if phase == "disrupt":
                meta["disrupt_start"] = len(paths)
            elif phase == "recover":
                meta["recover_start"] = len(paths)
            for rec, path in zip(by_phase[phase], frames_by[(scenario, phase)]):
                paths.append(path)
                gt.append({"present": None if rec.get("edge")
                           else bool(rec["present"])})
        episodes[scenario] = {"frames": paths, "gt": gt, "meta": meta}
    return episodes, skipped


KIND = {
    "occlusion_full": "occlusion",
    "occlusion_remove": "occlusion",
    "impostor": "occlusion",
    "identity_same_class": "occlusion",
    "lighting": "environment",
    "distractor": "unexpected",
    "occlude_book": "occlusion",
    "occlude_paper": "occlusion",
    "occlude_box": "occlusion",
    "occlude_jacket": "occlusion",
    "camera_pose": "environment",
}


def main():
    from run_study import CONDITIONS, score_episode, make_fast_verifier
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    target = " ".join(args).strip() or "red cup"
    want = [c for c in sys.argv[1:] if c in CONDITIONS]
    if want:
        target = "red cup"

    episodes, skipped = build_episodes()
    if not episodes:
        raise SystemExit("ABORT: no usable episodes")

    scored = sum(1 for e in episodes.values()
                 for g in e["gt"] if g["present"] is not None)
    total = sum(len(e["frames"]) for e in episodes.values())
    print(f"target={target!r}  episodes={len(episodes)}  frames={total}  "
          f"scored={scored} ({total - scored} edge-excluded)")
    for s in skipped:
        print(f"  SKIPPED {s}")
    print(f"\n!!  n=1 per scenario. Qualitative failures only -- do NOT rank "
          f"conditions\n    on small gaps. See the module docstring.\n")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)

    print("loading verifier...")
    verifier = make_fast_verifier()
    probe_v = getattr(verifier, "generate", None)
    if probe_v is None:
        print("  !! no .generate -- F_probe cannot run a real probe; "
              "skipping it")

    conds = want or list(CONDITIONS)
    all_res = {}
    for cond in conds:
        cfg = dict(CONDITIONS[cond])
        if cfg.get("use_region_probe") and probe_v is None:
            continue
        print(f"=== {cond} ===")
        all_res[cond] = {}
        for name, ep in sorted(episodes.items()):
            pipe = RecoveryPipeline(
                target, tracker=tracker,
                verifier=verifier if cfg.get("use_vlm") else None,
                probe_verifier=probe_v if cfg.get("use_region_probe") else None,
                **cfg)
            r = score_episode(pipe, ep, ep["frames"])
            r["states"] = dict(Counter(x["status"] for x in pipe.log))
            all_res[cond][name] = r
            print(f"  {name:<20} durAcc={str(r['during_disruption_acc']):<6} "
                  f"lost={r['lost_while_present']:<4} "
                  f"false={r['false_belief_frames']:<4} "
                  f"spur={str(r['spurious_reacquisition']):<5} "
                  f"vlm={r['vlm_calls']}")

    print("\n" + "=" * 78)
    print("LIVE CORPUS -- durAcc / lost / false, BY KIND")
    print("=" * 78)
    kinds = sorted({KIND.get(s, "?") for s in episodes})
    print(f"{'condition':<14}" + "".join(f"{k:>17}" for k in kinds))
    print("-" * 78)
    by_kind = {}
    for cond, eps in all_res.items():
        by_kind[cond] = {}
        row = f"{cond:<14}"
        for k in kinds:
            sel = [e for n, e in eps.items() if KIND.get(n) == k]
            accs = [e["during_disruption_acc"] for e in sel
                    if e["during_disruption_acc"] is not None]
            da = round(sum(accs) / len(accs), 3) if accs else None
            lost = sum(e["lost_while_present"] for e in sel)
            fb = sum(e["false_belief_frames"] for e in sel)
            by_kind[cond][k] = {"durAcc": da, "lost": lost, "false": fb,
                                "n": len(sel)}
            row += f"{str(da) + '/' + str(lost) + '/' + str(fb):>17}"
        print(row)

    if "D_full" in all_res and "F_probe" in all_res:
        print("\n" + "=" * 78)
        print("D vs F ON REAL OCCLUSION -- the comparison the synthetic "
              "corpus inverts")
        print("=" * 78)
        d, f = by_kind["D_full"].get("occlusion"), by_kind["F_probe"].get(
            "occlusion")
        if d and f:
            print(f"  synthetic: D durAcc 0.968 lost 4   |  "
                  f"F durAcc 0.786 lost 24   -> D wins")
            print(f"  LIVE:      D durAcc {d['durAcc']} lost {d['lost']}   |  "
                  f"F durAcc {f['durAcc']} lost {f['lost']}")
            if f["durAcc"] is not None and d["durAcc"] is not None:
                if f["durAcc"] > d["durAcc"]:
                    print("\n  The ordering REVERSES on real frames. The probe "
                          "is not a worse\n  mechanism -- it is a mechanism the "
                          "synthetic corpus cannot score,\n  because naming a "
                          "pasted rectangle is not the task it does.")
                else:
                    print("\n  The ordering HOLDS on real frames too. That "
                          "weakens the case for\n  the probe considerably: it "
                          "would mean F's synthetic loss is a real\n  loss, not "
                          "a corpus artifact. Report it that way.")

    json.dump({"by_kind": by_kind, "per_episode": all_res}, open(OUT, "w"),
              indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
