import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(13, 6.5))
stages = [
    ("Aug 6\nCycle 1 baseline\n(per-object VLM,\ndegenerate bug)", 65900,
     "#b33"),
    ("Aug 7\n--image-min-tokens\nremoved (GPU fix)", 3000, "#d70"),
    ("Aug 8\nopen listing\n(1 call, all objects)", 2740, "#d70"),
    ("Aug 8\nmajority-of-3\nvoting", 3500, "#d70"),
    ("Aug 8\nscene gating\n(stationary robot)", 830, "#390"),
    ("Aug 8\nVLM-configures +\nOpenCV tracks", 8, "#06c"),
]
xs = np.arange(len(stages))
vals = [s[1] for s in stages]
cols = [s[2] for s in stages]
bars = ax.bar(xs, vals, color=cols)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([s[0] for s in stages], fontsize=9)
ax.set_ylabel("ms per frame (log scale)")
ax.set_title("Detection latency per frame across the project — 65.9 s → 8 ms "
             "(≈8,200×)", fontsize=13, fontweight="bold")
for b, v in zip(bars, vals):
    lbl = f"{v/1000:.1f} s" if v >= 1000 else f"{v} ms"
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, lbl, ha="center",
            fontweight="bold", fontsize=10)
ax.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("report_latency_evolution.png", dpi=120)
print("wrote report_latency_evolution.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))

labels = ["per-object\nprompt", "open listing\n(single)",
          "open listing\n+ majority-3"]
acc = [4 / 8, 0.625, 0.771]
b = a1.bar(labels, acc, color=["#b33", "#d70", "#390"])
a1.set_ylim(0, 1)
a1.set_ylabel("held-out accuracy")
a1.set_title("Held-out accuracy (frames never tuned on)", fontweight="bold")
for bb, v in zip(b, acc):
    a1.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontweight="bold")
a1.grid(alpha=.3, axis="y")

h_labels = ["per-object prompt\n(bicycle ×3, banana ×2)",
            "open listing\n(all variants)"]
hal = [6, 0]
b2 = a2.bar(h_labels, hal, color=["#b33", "#390"])
a2.set_ylabel("hallucinated objects / 40 held-out probes")
a2.set_title("Hallucinations: naming the target vs open listing",
             fontweight="bold")
for bb, v in zip(b2, hal):
    a2.text(bb.get_x() + bb.get_width() / 2, v + 0.1, str(v), ha="center",
            fontweight="bold", fontsize=14)
a2.set_ylim(0, 7)
a2.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("report_accuracy_evolution.png", dpi=120)
print("wrote report_accuracy_evolution.png")
