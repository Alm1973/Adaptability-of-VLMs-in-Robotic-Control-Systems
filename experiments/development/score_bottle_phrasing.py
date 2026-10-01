
import json
import statistics as st
import sys

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "bottle_phrasing_results.json"

CONTROL = "water bottle"
CYCLE1_FN = {1, 2, 8}
NEGATIVES = {7, 9, 10}


def score_one(rows):
    tp = sum(1 for r in rows if r["gt"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in rows if r["gt"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in rows if r["gt"] == 0 and r["pred"] != 0)
    fn = sum(1 for r in rows if r["gt"] == 1 and r["pred"] != 1)
    npos, nneg = tp + fn, tn + fp
    sens = tp / npos if npos else None
    spec = tn / nneg if nneg else None
    bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
    fn_frames = sorted(r["idx"] for r in rows if r["gt"] == 1 and r["pred"] != 1)
    fp_frames = sorted(r["idx"] for r in rows if r["gt"] == 0 and r["pred"] != 0)
    none_n = sum(1 for r in rows if r["pred"] is None)
    lat = st.mean(r["latency_s"] for r in rows) if rows else 0
    return dict(sens=sens, spec=spec, bal=bal, fn=fn_frames, fp=fp_frames,
                none_n=none_n, lat=lat, npos=npos, nneg=nneg)


def pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    rows = json.load(open(RESULTS_FILE))
    print(f"scoring: {RESULTS_FILE}")
    phrasings = []
    for r in rows:
        if r["phrasing"] not in phrasings:
            phrasings.append(r["phrasing"])

    print(f"loaded {len(rows)} results\n")
    hdr = (f"{'phrasing':<22} {'sens':<7} {'spec':<7} {'bal':<7} "
           f"{'FN frames':<16} {'FP frames':<12} {'recovered':<12} none")
    print(hdr)
    print("-" * len(hdr))

    scored = {}
    for p in phrasings:
        s = score_one([r for r in rows if r["phrasing"] == p])
        scored[p] = s
        recovered = sorted(CYCLE1_FN - set(s["fn"]))
        print(f"{p:<22} {pct(s['sens']):<7} {pct(s['spec']):<7} "
              f"{pct(s['bal']):<7} {str(s['fn']):<16} {str(s['fp']):<12} "
              f"{str(recovered):<12} {s['none_n']}")

    ctrl = scored.get(CONTROL)
    print(f"\nControl = {CONTROL!r}: sens {pct(ctrl['sens'])}, "
          f"spec {pct(ctrl['spec'])}, bal {pct(ctrl['bal'])}, "
          f"FN {ctrl['fn']}")
    print("\nCLEAN WIN = recovers >=1 control-FN frame AND spec not below "
          "control (no new FP on 7/9/10):")
    any_win = False
    for p in phrasings:
        if p == CONTROL:
            continue
        s = scored[p]
        recovered = set(ctrl["fn"]) - set(s["fn"])
        new_fp = set(s["fp"]) - set(ctrl["fp"])
        clean = bool(recovered) and not new_fp
        tag = "  <-- CLEAN WIN" if clean else ""
        any_win = any_win or clean
        note = []
        if recovered:
            note.append(f"recovers {sorted(recovered)}")
        if new_fp:
            note.append(f"NEW FP {sorted(new_fp)}")
        if not note:
            note.append("no change vs control")
        print(f"  {p:<22} {'; '.join(note)}{tag}")
    print(f"\nHYPOTHESIS {'SUPPORTED' if any_win else 'NOT SUPPORTED'} "
          f"(>=1 clean-win phrasing exists: {any_win})")

    lat = [r["latency_s"] for r in rows]
    print(f"\nLatency (s): mean={st.mean(lat):.1f} "
          f"median={st.median(lat):.1f} max={max(lat):.1f}")


if __name__ == "__main__":
    main()
