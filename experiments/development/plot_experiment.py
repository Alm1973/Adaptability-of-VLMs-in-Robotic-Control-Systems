import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "live_experiment.json"
OUT = "live_experiment_chart.png"


def main():
    if not os.path.exists(SRC):
        print(f"{SRC} not found -- run live_experiment.py first")
        return
    rec = json.load(open(SRC))
    if not rec:
        print("no records")
        return

    new_ms = np.array([r["new_total_s"] * 1000 for r in rec])
    base_ms = np.array([r["baseline_s"] * 1000 for r in rec])
    cv_ms = np.array([r["new_opencv_ms"] for r in rec])
    targets = sorted({r["target"] for r in rec})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Live servo experiment: VLM-every-frame vs "
                 "VLM-configures-then-verifies\n"
                 f"{len(rec)} paired observations from real arm poses "
                 f"(GPU placement)", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    ax.boxplot([base_ms, new_ms, cv_ms],
               tick_labels=["baseline\n(VLM/frame)", "new\n(total)",
                            "new\n(OpenCV only)"])
    ax.set_yscale("log")
    ax.set_ylabel("ms per frame (log)")
    ax.set_title("Per-frame latency")
    ax.grid(alpha=.3, axis="y")
    ax.text(0.02, 0.02,
            f"median: {np.median(base_ms):.0f} ms  ->  "
            f"{np.median(cv_ms):.1f} ms\n"
            f"= {np.median(base_ms)/max(np.median(cv_ms),1e-9):.0f}x faster "
            f"on OpenCV-only frames",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(fc="lightyellow", alpha=.8))

    ax = axes[0][1]
    x = np.arange(len(targets))
    bw = 0.35
    bmeans = [np.mean([r["baseline_s"] * 1000 for r in rec
                       if r["target"] == t]) for t in targets]
    nmeans = [np.mean([r["new_total_s"] * 1000 for r in rec
                       if r["target"] == t]) for t in targets]
    ax.bar(x - bw/2, bmeans, bw, label="baseline", color="#c44")
    ax.bar(x + bw/2, nmeans, bw, label="new", color="#4a4")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15)
    ax.set_ylabel("mean ms (log)")
    ax.set_title("Latency by target")
    ax.legend()
    ax.grid(alpha=.3, axis="y")

    ax = axes[0][2]
    nb = np.mean([r["baseline_calls"] for r in rec])
    nn = np.mean([r["new_verify_calls"] for r in rec])
    bars = ax.bar(["baseline", "new"], [nb, nn], color=["#c44", "#4a4"])
    ax.set_ylabel("VLM calls per frame")
    ax.set_title("VLM invocations per frame")
    for b, v in zip(bars, [nb, nn]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    ax.grid(alpha=.3, axis="y")

    ax = axes[1][0]
    ax.plot(np.cumsum(base_ms) / 1000, label="baseline", color="#c44", lw=2)
    ax.plot(np.cumsum(new_ms) / 1000, label="new", color="#4a4", lw=2)
    ax.set_xlabel("frame index across the sweep")
    ax.set_ylabel("cumulative seconds")
    ax.set_title("Cumulative cost of the search episode")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[1][1]
    br = [np.mean([r["baseline_pred"] for r in rec if r["target"] == t])
          for t in targets]
    nr = [np.mean([r["new_pred"] for r in rec if r["target"] == t])
          for t in targets]
    ag = [np.mean([r["agree"] for r in rec if r["target"] == t])
          for t in targets]
    ax.bar(x - bw/2, br, bw, label="baseline detect rate", color="#c44")
    ax.bar(x + bw/2, nr, bw, label="new detect rate", color="#4a4")
    ax.plot(x, ag, "ko--", label="agreement", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of poses")
    ax.set_title("Detection rate & agreement\n(NOT accuracy - frames unlabelled)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1][2]
    by_cue = defaultdict(list)
    for r in rec:
        key = "+".join(r["cues"]) if r["cues"] else "(config failed)"
        by_cue[key].append(r["new_opencv_ms"])
    keys = sorted(by_cue, key=lambda k: -np.mean(by_cue[k]))
    ax.barh(range(len(keys)), [np.mean(by_cue[k]) for k in keys],
            color="#48a")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel("mean OpenCV ms")
    ax.set_title("OpenCV cost by cue set\n(edge proposals cost more than HSV)")
    ax.grid(alpha=.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")

    print("\n===== SUMMARY =====")
    print(f"observations      : {len(rec)}")
    print(f"baseline median   : {np.median(base_ms):8.1f} ms/frame")
    print(f"new total median  : {np.median(new_ms):8.1f} ms/frame")
    print(f"new OpenCV median : {np.median(cv_ms):8.1f} ms/frame")
    print(f"speedup (opencv)  : {np.median(base_ms)/max(np.median(cv_ms),1e-9):8.0f}x")
    print(f"VLM calls/frame   : {nb:.2f} -> {nn:.2f}")
    print(f"total wall time   : baseline {base_ms.sum()/1000:.1f}s  "
          f"new {new_ms.sum()/1000:.1f}s")
    print(f"method agreement  : {np.mean([r['agree'] for r in rec]):.1%}")


if __name__ == "__main__":
    main()
