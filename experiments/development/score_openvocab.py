
import json
import statistics as st


def pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    rows = json.load(open("openvocab_results.json"))
    objs = []
    for r in rows:
        if r["obj"] not in objs:
            objs.append(r["obj"])

    print(f"loaded {len(rows)} results\n")
    print(f"{'object':<15} {'P/N':<7} {'sens':<7} {'spec':<7} {'bal_acc':<8} "
          f"{'FP frames':<20} {'FN frames'}")
    bal_accs = []
    all_neg = all_fp = 0
    for obj in objs:
        rs = [r for r in rows if r["obj"] == obj]
        tp = sum(1 for r in rs if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in rs if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn = sum(1 for r in rs if r["gt"] == 1 and r["pred"] != 1)
        npos = tp + fn
        nneg = tn + fp
        all_neg += nneg
        all_fp += fp
        sens = tp / npos if npos else None
        spec = tn / nneg if nneg else None
        if sens is not None and spec is not None:
            bal = (sens + spec) / 2
            bal_accs.append(bal)
        else:
            bal = None
        fp_frames = sorted(r["idx"] for r in rs if r["gt"] == 0 and r["pred"] != 0)
        fn_frames = sorted(r["idx"] for r in rs if r["gt"] == 1 and r["pred"] != 1)
        print(f"{obj:<15} {npos}/{nneg:<5} {pct(sens):<7} {pct(spec):<7} "
              f"{pct(bal):<8} {str(fp_frames):<20} {fn_frames}")

    macro = st.mean(bal_accs) if bal_accs else None
    overall_fp_rate = all_fp / all_neg if all_neg else None
    print(f"\nMACRO balanced accuracy (objects w/ both classes): {pct(macro)}")
    print(f"Overall FP rate (fp / all negatives): {pct(overall_fp_rate)} "
          f"({all_fp}/{all_neg})")

    lat = [r["latency_s"] for r in rows]
    gpu = [r["latency_s"] for r in rows if r["placement"] == "gpu"]
    cpu_n = sum(1 for r in rows if r["placement"] == "cpu_fallback")
    none_n = sum(1 for r in rows if r["pred"] is None)
    lat_sorted = sorted(lat)
    p90 = lat_sorted[int(0.9 * (len(lat_sorted) - 1))]
    print(f"\nLatency (s): mean={st.mean(lat):.1f} median={st.median(lat):.1f} "
          f"p90={p90:.1f}  GPU-only mean={st.mean(gpu):.1f} (n={len(gpu)})")
    print(f"CPU fallbacks: {cpu_n}   unparseable/degenerate answers: {none_n}")


if __name__ == "__main__":
    main()
