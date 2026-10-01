
import json
import statistics as st


def pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def score(rows):
    objs = []
    for r in rows:
        if r["obj"] not in objs:
            objs.append(r["obj"])
    per = {}
    bal_accs = []
    all_neg = all_fp = 0
    for obj in objs:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos, nneg = tp + fn, tn + fp
        all_neg += nneg
        all_fp += fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        bal = (sens + spec) / 2 if (sens is not None and spec is not None) else None
        if bal is not None:
            bal_accs.append(bal)
        per[obj] = {
            "npos": npos, "nneg": nneg, "sens": sens, "spec": spec, "bal": bal,
            "fp_frames": sorted(r["idx"] for r in rs if r["gt"] == 0 and r["pred"] != 0),
            "fn_frames": sorted(r["idx"] for r in rs if r["gt"] == 1 and r["pred"] != 1),
        }
    macro = st.mean(bal_accs) if bal_accs else None
    overall_fp = all_fp / all_neg if all_neg else None
    none_n = sum(1 for r in rows if r["pred"] is None)
    return per, macro, overall_fp, (all_fp, all_neg), none_n


def per_frame_latency(rows, batched):
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["idx"], [])
        by_frame[r["idx"]].append(r)
    frame_lat = []
    for idx, rs in by_frame.items():
        if batched:
            frame_lat.append(rs[0].get("call_latency_s", rs[0]["latency_s"]))
        else:
            frame_lat.append(sum(r["latency_s"] for r in rs))
    return frame_lat


def report(name, rows, batched):
    per, macro, overall_fp, (fp, neg), none_n = score(rows)
    print(f"\n===== {name} =====")
    print(f"{'object':<15} {'P/N':<7} {'sens':<7} {'spec':<7} {'bal_acc':<8} "
          f"{'FP frames':<16} {'FN frames'}")
    for obj, m in per.items():
        print(f"{obj:<15} {m['npos']}/{m['nneg']:<5} {pct(m['sens']):<7} "
              f"{pct(m['spec']):<7} {pct(m['bal']):<8} "
              f"{str(m['fp_frames']):<16} {m['fn_frames']}")
    print(f"MACRO balanced accuracy: {pct(macro)}")
    print(f"Overall FP rate: {pct(overall_fp)} ({fp}/{neg})")
    print(f"Unparseable/degenerate final answers: {none_n}")
    fl = per_frame_latency(rows, batched)
    print(f"Per-FRAME latency (s): mean={st.mean(fl):.1f} "
          f"median={st.median(fl):.1f} min={min(fl):.1f} max={max(fl):.1f} "
          f"total={sum(fl):.1f}  (n_frames={len(fl)})")
    return macro, overall_fp


def main():
    print("Cycle 2 comparison: batched (1 call/frame) vs per-object (Cycle 1)")
    try:
        base = json.load(open("openvocab_results.json"))
        b_macro, b_fp = report("PER-OBJECT (Cycle 1 baseline)", base, batched=False)
    except FileNotFoundError:
        b_macro = b_fp = None
        print("(openvocab_results.json not found -- baseline skipped)")
    batch = json.load(open("openvocab_batched_results.json"))
    x_macro, x_fp = report("BATCHED (Cycle 2)", batch, batched=True)

    if b_macro is not None:
        print("\n===== VERDICT =====")
        print(f"MACRO bal acc:  per-object {pct(b_macro)}  ->  batched {pct(x_macro)} "
              f"(delta {x_macro - b_macro:+.3f})")
        print(f"Overall FP rate: per-object {pct(b_fp)}  ->  batched {pct(x_fp)} "
              f"(delta {x_fp - b_fp:+.3f})")


if __name__ == "__main__":
    main()
