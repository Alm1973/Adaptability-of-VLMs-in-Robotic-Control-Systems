#!/usr/bin/env python3
"""
analyze.py -- regenerate every number, table and figure in the AVI paper
from the raw files of the pre-registered controlled-rig run 2026-09-01_233533.

Usage (from the repository root):
    python results/analysis/analyze.py --run results/2026-09-01_final \
                                       --videos media/videos --out results

Needs ffmpeg on PATH (used to pull single frames from the videos).

Inputs (read-only):
    results.csv            live-run summary, one row per trial
    results.json           live-run per-frame belief trajectory (8,428 frames)
    ablation_scores.json   offline replay of the same frames: D_full, E_no_vlm,
                           and D_full with decay 12 / 30 / 60
    baseline/*.jpg         bare-rig reference frames
    media/videos/*.mp4     annotated 10 fps clips (for the qualitative figure)

Conventions (see the paper, Section 5):
  * The OFFLINE REPLAY (ablation_scores.json) is authoritative for every
    D_full-vs-E_no_vlm comparison: both conditions read byte-identical frames.
  * "during-disruption accuracy" = disrupt phase only, edge frames excluded
    (the replay scorer). The live scorer also counted the recover phase; that
    lenient number is reported separately as "episode accuracy (live)".
  * False belief is never folded into accuracy.
"""
import argparse
import tempfile
import csv
import json
import math
import os
import subprocess
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
})

SCENARIOS = ["control", "lighting", "camera_pose", "occlude_hand",
             "occlude_object", "removal", "substitution", "identity_swap"]
ABSENT_ENDING = ["removal", "substitution", "identity_swap"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def verdict(acc, false):
    if false == 0 and acc >= 0.7:
        return "PASS"
    if false == 0 and acc >= 0.4:
        return "PARTIAL"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, "figures")
    tab_dir = os.path.join(a.out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(a.run, "results.csv"))))
    trials = json.load(open(os.path.join(a.run, "results.json")))
    abl = json.load(open(os.path.join(a.run, "ablation_scores.json")))["per_episode"]
    keys = sorted(abl["D_full"].keys())          # '001_camera_pose_rep1', ...
    assert len(rows) == len(trials) == len(keys) == 48
    D, E = abl["D_full"], abl["E_no_vlm"]
    S = {}                                        # summary numbers for the paper

    # ------------------------------------------------ corpus & timing
    frames = [f for t in trials for f in t["frames"]]
    S["frames_total"] = len(frames)
    S["frames_scored"] = sum(not f["edge"] for f in frames)
    S["frames_edge"] = sum(f["edge"] for f in frames)
    S["frames_absent_scored"] = sum((not f["edge"]) and (not f["present"]) for f in frames)
    # 't' restarts at each phase, so take intervals within a phase only
    gaps = []
    for t in trials:
        fr = t["frames"]
        gaps += [b["t"] - a["t"] for a, b in zip(fr, fr[1:]) if a["phase"] == b["phase"]]
    gaps = np.array(gaps)
    S["capture_dt_median_s"] = float(np.median(gaps))
    S["capture_dt_mean_s"] = float(np.mean(gaps))
    S["capture_fps_median"] = 1 / S["capture_dt_median_s"]
    S["capture_fps_mean"] = 1 / S["capture_dt_mean_s"]
    S["trial_minutes"] = sum(float(r["wall_clock_s"]) for r in rows) / 60

    # VLM frames: the per-frame vlm_calls field is cumulative within a trial
    vlm_ms, novlm_ms = [], []
    live_calls = 0
    for t in trials:
        prev = 0
        for f in t["frames"]:
            if f["vlm_calls"] > prev:
                vlm_ms.append(f["step_ms"])
            else:
                novlm_ms.append(f["step_ms"])
            prev = f["vlm_calls"]
        live_calls += t["frames"][-1]["vlm_calls"]
    vlm_ms, novlm_ms = np.array(vlm_ms), np.array(novlm_ms)
    S["vlm_frames"] = int(len(vlm_ms))
    S["vlm_calls_live"] = int(live_calls)
    S["vlm_frame_share_pct"] = 100 * len(vlm_ms) / len(frames)
    S["step_ms_novlm_median"] = float(np.median(novlm_ms))
    S["step_ms_novlm_p95"] = float(np.percentile(novlm_ms, 95))
    S["step_ms_vlm_median"] = float(np.median(vlm_ms))
    S["step_ms_vlm_p95"] = float(np.percentile(vlm_ms, 95))
    S["step_ms_all_mean"] = float(np.mean(np.r_[vlm_ms, novlm_ms]))
    S["vlm_calls_replay_D"] = int(sum(D[k]["vlm_calls"] for k in keys))

    # ------------------------------------------------ verdicts
    live_v = Counter(r["verdict"] for r in rows)
    corr_v = Counter(verdict(D[k]["durAcc"], D[k]["false"]) for k in keys)
    S["verdicts_live"] = dict(live_v)
    S["verdicts_replay_disrupt_only"] = dict(corr_v)
    vb_rows = []
    for cond in ["D_full", "E_no_vlm"]:
        c = Counter()
        for k in keys:
            v = verdict(abl[cond][k]["durAcc"], abl[cond][k]["false"])
            if v == "FAIL":
                v = "FAIL (false belief)" if abl[cond][k]["false"] > 0 else "FAIL (accuracy < 0.4, no false belief)"
            c[v] += 1
        vb_rows.append({"condition": cond, **{k2: c.get(k2, 0) for k2 in
                        ["PASS", "PARTIAL", "FAIL (false belief)", "FAIL (accuracy < 0.4, no false belief)"]}})
    write_csv(os.path.join(tab_dir, "verdicts_preregistered_rule.csv"), vb_rows)
    S["verdict_breakdown"] = vb_rows
    S["live_false_total"] = sum(int(r["false_belief_frames"]) for r in rows)
    S["live_episode_acc_mean"] = float(np.mean([float(r["during_disruption_acc"]) for r in rows]))

    # ------------------------------------------------ ablation
    def tot(cond, field):
        return sum(abl[cond][k][field] for k in keys)

    abl_rows = []
    for cond in ["D_full", "E_no_vlm", "D_full(decay=30)", "D_full(decay=60)"]:
        abl_rows.append({
            "condition": cond,
            "false_belief": tot(cond, "false"),
            "lost_while_present": tot(cond, "lost"),
            "spurious_reacq": sum(bool(abl[cond][k]["spurious"]) for k in keys),
            "mean_durAcc": round(float(np.mean([abl[cond][k]["durAcc"] for k in keys])), 3),
            "vlm_calls": tot(cond, "vlm_calls"),
        })
    write_csv(os.path.join(tab_dir, "ablation_summary.csv"), abl_rows)
    S["ablation"] = abl_rows

    diffs = []
    for k in keys:
        d = E[k]["false"] - D[k]["false"]
        if d != 0:
            clutter = rows[keys.index(k)]["clutter"]
            diffs.append({"episode": k, "clutter": clutter,
                          "D_false": D[k]["false"], "E_false": E[k]["false"],
                          "E_minus_D": d})
    write_csv(os.path.join(tab_dir, "differing_trials.csv"), diffs)
    n_diff = len(diffs)
    n_pos = sum(x["E_minus_D"] > 0 for x in diffs)
    S["n_differing"] = n_diff
    S["n_favour_vlm"] = n_pos
    S["sign_p_one_sided"] = float(stats.binomtest(n_pos, n_diff, 0.5, alternative="greater").pvalue)
    S["sign_p_two_sided"] = float(stats.binomtest(n_pos, n_diff, 0.5).pvalue)
    S["frames_saved_total"] = sum(x["E_minus_D"] for x in diffs)
    top2 = sorted((x["E_minus_D"] for x in diffs), reverse=True)[:2]
    S["top2_share_pct"] = 100 * sum(top2) / S["frames_saved_total"]

    # by scenario and by clutter (replay)
    scen_rows = []
    for sc in SCENARIOS:
        ks = [k for k in keys if D[k]["scenario"] == sc]
        for cl in ["none", "high"]:
            kk = [k for k in ks if rows[keys.index(k)]["clutter"] == cl]
            scen_rows.append({
                "scenario": sc, "clutter": cl, "n": len(kk),
                "D_false": sum(D[k]["false"] for k in kk),
                "E_false": sum(E[k]["false"] for k in kk),
                "D_durAcc_mean": round(float(np.mean([D[k]["durAcc"] for k in kk])), 3),
                "E_durAcc_mean": round(float(np.mean([E[k]["durAcc"] for k in kk])), 3),
                "D_vlm_calls": sum(D[k]["vlm_calls"] for k in kk),
                "verdicts_replay": " ".join(verdict(D[k]["durAcc"], D[k]["false"])[0:4] for k in kk),
            })
    write_csv(os.path.join(tab_dir, "by_scenario_clutter.csv"), scen_rows)
    S["by_clutter"] = {}
    for cl in ["none", "high"]:
        kk = [k for k in keys if rows[keys.index(k)]["clutter"] == cl]
        S["by_clutter"][cl] = {"D_false": sum(D[k]["false"] for k in kk),
                               "E_false": sum(E[k]["false"] for k in kk)}
    ident = [k for k in keys if D[k]["scenario"] == "identity_swap"]
    S["identity"] = {"D_false": sum(D[k]["false"] for k in ident),
                     "E_false": sum(E[k]["false"] for k in ident),
                     "D_acc": float(np.mean([D[k]["durAcc"] for k in ident])),
                     "E_acc": float(np.mean([E[k]["durAcc"] for k in ident]))}

    # ------------------------------------------------ detector false positives on absent frames (live)
    det_rows = []
    fp_conf = []
    for sc in ABSENT_ENDING + ["ALL"]:
        for cl in ["none", "high"]:
            k_abs = k_det = 0
            for r, t in zip(rows, trials):
                if r["clutter"] != cl or (sc != "ALL" and r["scenario"] != sc):
                    continue
                for f in t["frames"]:
                    if not f["edge"] and not f["present"]:
                        k_abs += 1
                        if f["detected"]:
                            k_det += 1
                            if sc == "ALL":
                                fp_conf.append(f["detector_conf"])
            lo, hi = wilson(k_det, k_abs)
            det_rows.append({"scenario": sc, "clutter": cl, "absent_frames": k_abs,
                             "detector_fired": k_det,
                             "rate": round(k_det / k_abs, 4) if k_abs else None,
                             "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    write_csv(os.path.join(tab_dir, "detector_fp_absent.csv"), det_rows)
    allr = {r["clutter"]: r for r in det_rows if r["scenario"] == "ALL"}
    table = [[allr["none"]["detector_fired"], allr["none"]["absent_frames"] - allr["none"]["detector_fired"]],
             [allr["high"]["detector_fired"], allr["high"]["absent_frames"] - allr["high"]["detector_fired"]]]
    S["detector_fp"] = det_rows
    S["fisher_p_bare_vs_clutter"] = float(stats.fisher_exact(table)[1])
    S["fp_conf_median"] = float(np.median(fp_conf))
    S["fp_conf_min"] = float(np.min(fp_conf))
    S["fp_conf_max"] = float(np.max(fp_conf))

    fpc = np.array(fp_conf)
    S["fp_conf_share_ge_025"] = float(np.mean(fpc >= 0.25))
    S["fp_conf_share_ge_050"] = float(np.mean(fpc >= 0.50))

    # decay sweep split by axis, and VLM calls per episode by axis (replay)
    axis_rows = []
    for cond in ["D_full", "D_full(decay=30)", "D_full(decay=60)", "E_no_vlm"]:
        for ax_name in ["control", "environment", "occlusion", "identity", "unexpected"]:
            kk = [k for k in keys if abl[cond][k]["axis"] == ax_name]
            axis_rows.append({"condition": cond, "axis": ax_name, "n": len(kk),
                              "false_belief": sum(abl[cond][k]["false"] for k in kk),
                              "lost": sum(abl[cond][k]["lost"] for k in kk),
                              "mean_durAcc": round(float(np.mean([abl[cond][k]["durAcc"] for k in kk])), 3),
                              "vlm_calls_per_episode": round(sum(abl[cond][k]["vlm_calls"] for k in kk) / len(kk), 2)})
    write_csv(os.path.join(tab_dir, "by_axis_conditions.csv"), axis_rows)
    S["by_axis"] = axis_rows

    # ------------------------------------------------ occlusion trials: what held belief?
    occ_rows = []
    for r, t in zip(rows, trials):
        if r["scenario"] not in ("occlude_hand", "occlude_object"):
            continue
        dis = [f for f in t["frames"] if not f["edge"] and f["phase"] == "disrupt"]
        st = Counter(f["status"] for f in dis)
        occ_rows.append({
            "trial": int(r["trial_no"]), "scenario": r["scenario"], "clutter": r["clutter"],
            "disrupt_frames": len(dis),
            "belief_correct": round(sum(f["believes"] == f["present"] for f in dis) / len(dis), 3),
            "detector_fired": round(sum(bool(f["detected"]) for f in dis) / len(dis), 3),
            "CONFIRMED": st.get("CONFIRMED", 0), "OCCLUDED": st.get("OCCLUDED", 0),
            "MISSING": st.get("MISSING", 0), "DISPLACED": st.get("DISPLACED", 0),
            "AMBIGUOUS": st.get("AMBIGUOUS", 0)})
    write_csv(os.path.join(tab_dir, "occlusion_disrupt_states.csv"), occ_rows)
    S["occlusion"] = occ_rows

    # detection rate after the target left (absent-ending scenarios): the staging confound
    conf_rows = []
    for r, t in zip(rows, trials):
        if r["scenario"] not in ABSENT_ENDING:
            continue
        ab = [f for f in t["frames"] if not f["edge"] and not f["present"]]
        k = keys[int(r["trial_no"]) - 1]
        conf_rows.append({"trial": int(r["trial_no"]), "scenario": r["scenario"],
                          "clutter": r["clutter"], "absent_frames": len(ab),
                          "detector_fired_pct": round(100 * sum(bool(f["detected"]) for f in ab) / len(ab), 1),
                          "D_false": D[k]["false"], "E_false": E[k]["false"]})
    write_csv(os.path.join(tab_dir, "absent_detection_confound.csv"), conf_rows)

    # ------------------------------------------------ figures
    fig_pipeline(os.path.join(fig_dir, "fig1_pipeline.png"))
    fig_rig(a.run, a.videos, os.path.join(fig_dir, "fig2_rig.png"))
    fig_detector_fp(det_rows, os.path.join(fig_dir, "fig4_detector_fp.png"))
    fig_paired(keys, D, E, rows, os.path.join(fig_dir, "fig3_false_belief_per_trial.png"))
    fig_decay(abl_rows, os.path.join(fig_dir, "fig6_decay_tradeoff.png"))
    fig_latency(vlm_ms, novlm_ms, os.path.join(fig_dir, "fig7_latency.png"))
    fig_qualitative(a.videos, os.path.join(fig_dir, "fig5_qualitative.png"))

    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in S.items() if not isinstance(v, list)}, indent=2, default=float))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ================================================================ figures
def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=fc, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 6) / 2, sub, ha="center", va="center",
                fontsize=7, color=INK2, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                    shrinkA=0, shrinkB=0))

    def note(x, y, t, ha="center"):
        ax.text(x, y, t, ha=ha, va="center", fontsize=6.6, color=INK2, linespacing=1.25)

    # top row
    box(1, 42, 17, 18, "Camera", "Logitech C270\n10 fps", "#f0efec")
    box(30, 42, 22, 18, "Detector", "YOLO-World\n\"red cup\", every\nframe, ~14 ms", "#cde2fb")
    box(64, 40, 34, 22, "Belief state machine",
        "CONFIRMED  OCCLUDED\nDISPLACED  MISSING\nAMBIGUOUS\n12-frame decay timer", "#f0efec")
    box(118, 42, 20, 18, "Arm", "2 servos\nhomed 90/90", "#f0efec")
    # bottom row
    box(28, 3, 26, 18, "OpenCV geometry", "runs only on a\nlost frame, ~12 ms", "#d6f0e5")
    box(86, 3, 30, 18, "VLM verifier", "Qwen2.5-VL-3B, 4-bit\nyes/no on the box crop", "#fbdccf")
    # flows
    arrow(18.8, 51, 29.2, 51)
    arrow(52.8, 51, 63.2, 51)
    note(58, 55, "box +\nscore")
    arrow(98.8, 51, 117.2, 51)
    note(108, 57, "acts only if\nCONFIRMED")
    arrow(41, 41.2, 41, 21.8)
    note(43, 31, "no target\nbox", ha="left")
    arrow(54.8, 12, 70, 39.2)
    note(66, 22, "why was it\nlost? (diagnosis)", ha="left")
    arrow(90, 39.2, 96, 21.8)
    arrow(106, 21.8, 100, 39.2)
    note(118, 32, "asked only when a box\nreappears while the\nbelief is not CONFIRMED;\nyes -> CONFIRMED,\nno -> AMBIGUOUS", ha="left")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _video_frame(video, n):
    tmp = os.path.join(tempfile.gettempdir(), f"_vf_{os.path.basename(video)}_{n}.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-vf",
                    f"select=eq(n\\,{n})", "-frames:v", "1", tmp], check=True)
    return plt.imread(tmp)


def fig_rig(run, videos, path):
    base = sorted(p for p in os.listdir(os.path.join(run, "baseline")) if p.endswith(".jpg"))[0]
    im1 = plt.imread(os.path.join(run, "baseline", base))
    im2 = _video_frame(os.path.join(videos, "KEY_removal_high_both_fail_trial046.mp4"), 12)
    im2 = im2[int(im2.shape[0] * 0.08):]               # drop the overlay text strip
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, im, t in zip(axs, [im1, im2], ["(a) Bare rig: target only",
                                           "(b) High clutter: 6-8 objects incl. other cups"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=8.5, fontweight="normal", loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_detector_fp(det_rows, path):
    labels = ["removal", "substitution", "identity_swap", "ALL"]
    shown = ["removal", "substitution", "identity_swap", "all three"]
    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    for i, (cl, col, name) in enumerate([("none", BLUE, "bare rig"), ("high", ORANGE, "high clutter")]):
        vals, lo, hi, ns = [], [], [], []
        for s in labels:
            r = next(r for r in det_rows if r["scenario"] == s and r["clutter"] == cl)
            vals.append(100 * r["rate"])
            lo.append(100 * (r["rate"] - r["ci95_lo"]))
            hi.append(100 * (r["ci95_hi"] - r["rate"]))
            ns.append((r["detector_fired"], r["absent_frames"]))
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, width=w - 0.04, color=col, label=name, zorder=2)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8,
                    capsize=2, zorder=3)
        for xi, v, (k, n), e in zip(xs, vals, ns, hi):
            ax.text(xi, v + e + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, shown)
    ax.set_ylabel("Absent frames where the\ndetector still fired (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_paired(keys, D, E, rows, path):
    ks = [k for k in keys if D[k]["scenario"] in ABSENT_ENDING]
    ks.sort(key=lambda k: (ABSENT_ENDING.index(D[k]["scenario"]),
                           rows[keys.index(k)]["clutter"] != "none", k))
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(len(ks))
    for i, k in enumerate(ks):
        d, e = D[k]["false"], E[k]["false"]
        ax.plot([i, i], [d, e], color=MUTED if d == e else INK2, lw=1.0, zorder=1)
    ax.scatter(x, [E[k]["false"] for k in ks], s=30, color=ORANGE, edgecolor="white",
               linewidth=1.2, zorder=3, label="VLM removed (E_no_vlm)")
    ax.scatter(x, [D[k]["false"] for k in ks], s=30, color=BLUE, edgecolor="white",
               linewidth=1.2, zorder=4, label="Full system (D_full)")
    lab = []
    for k in ks:
        cl = rows[keys.index(k)]["clutter"]
        lab.append(f"{int(k[:3])}{'*' if cl == 'high' else ''}")
    ax.set_xticks(x, lab, fontsize=6.5)
    for j, s in enumerate(ABSENT_ENDING):
        ax.text(j * 6 + 2.5, 47, s, ha="center", fontsize=7.5, color=INK2)
        if j:
            ax.axvline(j * 6 - 0.5, color=GRID, lw=0.8)
    ax.set_ylim(-3, 50)
    ax.set_xlabel("Trial number (* = high clutter)")
    ax.set_ylabel("False-belief frames")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_decay(abl_rows, path):
    pts = [r for r in abl_rows if r["condition"].startswith("D_full")]
    lab = {"D_full": "12 frames (used)", "D_full(decay=30)": "30", "D_full(decay=60)": "60"}
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    xs = [r["lost_while_present"] for r in pts]
    ys = [r["false_belief"] for r in pts]
    ax.plot(xs, ys, color=BLUE, lw=2, zorder=2)
    ax.scatter(xs, ys, s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    for r, x, y in zip(pts, xs, ys):
        below = r["condition"] == "D_full"
        ax.text(x, y - 5 if below else y + 3.5,
                f"decay {lab[r['condition']]}\naccuracy {r['mean_durAcc']:.2f}",
                ha="center", va="top" if below else "bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Lost-while-present frames (safe error)")
    ax.set_ylabel("False-belief frames\n(unsafe error)")
    ax.set_xlim(150, 800)
    ax.set_ylim(170, 262)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_latency(vlm_ms, novlm_ms, path):
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    bins = np.logspace(np.log10(10), np.log10(3000), 60)
    ax.hist(novlm_ms, bins=bins, color=BLUE, alpha=0.85, label=f"no VLM call (n={len(novlm_ms):,})")
    ax.hist(vlm_ms, bins=bins, color=ORANGE, alpha=0.95, label=f"VLM call (n={len(vlm_ms):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ymax = ax.get_ylim()[1]
    for v, c in [(np.median(novlm_ms), BLUE), (np.median(vlm_ms), ORANGE)]:
        ax.axvline(v, color=INK, lw=0.9, ls="-")
        ax.text(v * 1.08, ymax * 1.6, f"median {v:.0f} ms", fontsize=7, color=INK,
                va="bottom")
    ax.set_ylim(top=ymax * 6)
    ax.set_xlabel("Processing time per frame (ms, log scale)")
    ax.set_ylabel("Frames (log scale)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_qualitative(videos, path):
    spec = [
        ("KEY_removal_high_both_fail_trial046.mp4",
         [(12, "baseline: red cup CONFIRMED"),
          (96, "hand covers cup: box jumps to\nthe blue mug, still CONFIRMED"),
          (156, "cup gone: still CONFIRMED\n= false belief, VLM never asked")],
         "Trial 46 (removal, clutter)"),
        ("KEY_removal_high_vlm_prevents_ghost_trial014.mp4",
         [(12, "baseline: red cup CONFIRMED"),
          (96, "cup hidden, box lands on the mug\nafter a lost frame: VLM says no"),
          (144, "cup gone: AMBIGUOUS, not\nbelieved present = correct")],
         "Trial 14 (removal, clutter)"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.3))
    for r, (vid, frames, rowname) in enumerate(spec):
        for c, (n, cap) in enumerate(frames):
            im = _video_frame(os.path.join(videos, vid), n)
            im = im[int(im.shape[0] * 0.07):]
            ax = axs[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(cap, fontsize=6.8, fontweight="normal", color=INK)
            if c == 0:
                ax.set_ylabel(rowname, fontsize=7.5, color=INK)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
