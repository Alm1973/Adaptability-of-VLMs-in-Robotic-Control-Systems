
import json
import sys

import cv2

import detect_methods as dm

GT = {
    1: True,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True,
    7: False,
    8: True,
    9: False,
    10: False,
    11: True,
    12: True,
}


def metrics(preds):
    tp = sum(1 for i, p in preds.items() if p and GT[i])
    tn = sum(1 for i, p in preds.items() if not p and not GT[i])
    fp = sum(1 for i, p in preds.items() if p and not GT[i])
    fn = sum(1 for i, p in preds.items() if not p and GT[i])
    npos = sum(1 for i in preds if GT[i])
    nneg = len(preds) - npos
    sens = tp / npos if npos else 0.0
    spec = tn / nneg if nneg else 0.0
    return {"correct": tp + tn, "n": len(preds), "tp": tp, "tn": tn,
            "fp": fp, "fn": fn, "sens": round(sens, 3), "spec": round(spec, 3),
            "balanced_acc": round((sens + spec) / 2, 3)}


def main():
    data = json.load(open("experiment_log.json"))
    log = data["log"]

    run_vlm = "--vlm" in sys.argv
    ctx = dm.build_context(cv2.imread("template_bottle.jpg"))

    names = ["t1", "t2", "t3", "t4", "t6"]
    preds = {n: {} for n in names}
    if run_vlm:
        preds["t5"] = {}

    print(f"{'pose':<26} {'GT':<5} " + " ".join(f"{n:<6}" for n in names) +
          ("  t5" if run_vlm else ""))
    for e in log:
        i = e["idx"]
        row = f"{e['label'][:24]:<26} {'YES' if GT[i] else 'no':<5} "
        for n in names:
            f = e["techniques"][n]["found"]
            preds[n][i] = f
            row += f"{'Y' if f else '.':<6} "
        if run_vlm:
            frame = cv2.imread(e["file"])
            vctx = dict(ctx)
            vctx["vlm_box"] = e["techniques"]["t1"].get("box")
            r = dm.t5_vlm(frame, vctx)
            preds["t5"][i] = r["found"]
            row += f"  {'Y' if r['found'] else '.'} ({r['meta'].get('raw')!r})"
        print(row)

    print(f"\n{'technique':<12} {'raw':<8} {'sens':<7} {'spec':<7} {'balanced':<9} notes")
    for n in preds:
        m = metrics(preds[n])
        note = ""
        if m["spec"] == 0.0 and m["sens"] > 0.9:
            note = "ALWAYS-YES: raw score is an artifact of class imbalance"
        if m["sens"] < 0.2 and m["spec"] > 0.9:
            note = "ALWAYS-NO: no real detection ability"
        print(f"{n:<12} {m['correct']}/{m['n']:<6} {m['sens']:<7} {m['spec']:<7} "
              f"{m['balanced_acc']:<9} {note}")

    print("\n--- failure detail ---")
    for n in names:
        fns = [i for i in preds[n] if GT[i] and not preds[n][i]]
        fps = [i for i in preds[n] if not GT[i] and preds[n][i]]
        print(f"{n}: FN(missed bottle)={fns}  FP(hallucinated bottle)={fps}")


if __name__ == "__main__":
    main()
