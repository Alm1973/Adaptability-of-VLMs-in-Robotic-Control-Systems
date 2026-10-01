import json

import open_vocab_detect as ov
from heldout_gt import HELDOUT_GT, HALLUCINATION_PROBES

YOLO_RAW = "yolo_world_raw.json"
VLM_RES = "crop_verify_results.json"
OUT = "hybrid_router_results.json"

QUERIES = ["laptop", "keyboard", "computer mouse", "water bottle"]
PROBES = list(HALLUCINATION_PROBES)

YOLO_UNRELIABLE = {"water bottle"}

CONF_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
YOLO_MODEL = "yolov8s-worldv2.pt"


def yolo_fired(yraw, frame, cls, conf):
    return any(d["cls"] == cls and d["conf"] >= conf
               for d in yraw[frame]["dets"])


def vlm_crops_union(vlm, frame, cls):
    lists = list(vlm[frame]["crops"].values())
    return any(ov.is_present(cls, l) for l in lists)


def vlm_whole_maj3(vlm, frame, cls):
    lists = vlm[frame]["whole"]
    return sum(1 for l in lists if ov.is_present(cls, l)) >= 2


def confusion(decide):
    tp = fp = tn = fn = 0
    per_class = {}
    for frame, labels in HELDOUT_GT.items():
        for cls, gt in labels.items():
            pred = decide(frame, cls)
            key = (int(gt), int(pred))
            per_class.setdefault(cls, []).append((frame, gt, pred))
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1
    p, n = tp + fn, tn + fp
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": round((tp + tn) / (p + n), 3) if (p + n) else None,
        "sens": round(tp / p, 3) if p else None,
        "spec": round(tn / n, 3) if n else None,
    }, per_class


def main():
    yraw = json.load(open(YOLO_RAW))[YOLO_MODEL]["raw"]
    vlm = json.load(open(VLM_RES))

    results = {}

    print("=" * 78)
    print("HYBRID ROUTER -- 16 held-out points (11 present / 5 absent, 4 frames)")
    print(f"YOLO model: {YOLO_MODEL}   |   VLM: crop_verify_results.json (run3)")
    print(f"routed to VLM: {sorted(YOLO_UNRELIABLE)}   rest -> YOLO")
    print("=" * 78)

    hdr = (f"{'condition':<30}{'acc':>6}{'sens':>7}{'spec':>7}"
           f"{'tp':>4}{'fp':>4}{'tn':>4}{'fn':>4}")

    print("\n[STANDALONE, for reference]")
    print(hdr); print("-" * len(hdr))
    for conf in CONF_SWEEP:
        m, _ = confusion(lambda f, c, cf=conf: yolo_fired(yraw, f, c, cf))
        results[f"yolo_only@{conf}"] = m
        print(f"{'yolo_only @'+str(conf):<30}{m['acc']:>6}{m['sens']:>7}"
              f"{m['spec']:>7}{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}")

    m, _ = confusion(lambda f, c: vlm_whole_maj3(vlm, f, c))
    results["vlm_whole_maj3(shipped)"] = m
    print(f"{'vlm_whole_maj3 (shipped)':<30}{m['acc']:>6}{m['sens']:>7}"
          f"{m['spec']:>7}{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}")
    m, _ = confusion(lambda f, c: vlm_crops_union(vlm, f, c))
    results["vlm_crops_union"] = m
    print(f"{'vlm_crops_union (all cls)':<30}{m['acc']:>6}{m['sens']:>7}"
          f"{m['spec']:>7}{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}")

    print("\n[HYBRID: bottle->VLM crops_union, rest->YOLO]")
    print(hdr); print("-" * len(hdr))

    def hybrid(frame, cls, conf):
        if cls in YOLO_UNRELIABLE:
            return vlm_crops_union(vlm, frame, cls)
        return yolo_fired(yraw, frame, cls, conf)

    best = None
    for conf in CONF_SWEEP:
        m, per_class = confusion(lambda f, c, cf=conf: hybrid(f, c, cf))
        results[f"hybrid@{conf}"] = m
        print(f"{'hybrid @'+str(conf):<30}{m['acc']:>6}{m['sens']:>7}"
              f"{m['spec']:>7}{m['tp']:>4}{m['fp']:>4}{m['tn']:>4}{m['fn']:>4}")
        if best is None or m["acc"] >= best[1]["acc"]:
            best = (conf, m, per_class)

    print("\n[THE TWO SYSTEMATIC ERRORS -- does the hybrid fix BOTH?]")
    yv = yolo_fired(yraw, "c2_09_tilt_b90_t108.jpg", "computer mouse", 0.25)
    vv = vlm_whole_maj3(vlm, "c2_09_tilt_b90_t108.jpg", "computer mouse")
    hv = hybrid("c2_09_tilt_b90_t108.jpg", "computer mouse", 0.25)
    print(f"  c2_09 mouse (gt=0, headphones): VLM={vv}  YOLO@0.25={yv}  "
          f"HYBRID={hv}  {'FIXED' if not hv else 'STILL WRONG'}")
    yb = yolo_fired(yraw, "c2_10_home.jpg", "water bottle", 0.10)
    hb = hybrid("c2_10_home.jpg", "water bottle", 0.10)
    print(f"  c2_10 bottle (gt=1, transparent): YOLO@0.10={yb}  "
          f"HYBRID={hb}  {'RECOVERED' if hb else 'STILL MISSED'}")

    conf, m, per_class = best
    print(f"\n[per-point, hybrid @ conf={conf}  acc={m['acc']}]")
    for cls in QUERIES:
        for frame, gt, pred in per_class[cls]:
            mark = "ok " if gt == pred else "XX "
            src = "VLM " if cls in YOLO_UNRELIABLE else "YOLO"
            print(f"  {mark}{src} {frame:<26} {cls:<15} gt={gt} pred={int(pred)}")

    probe_hits = []
    for frame in HELDOUT_GT:
        for probe in PROBES:
            if any(ov.is_present(probe, l)
                   for l in list(vlm[frame]["crops"].values())
                   + vlm[frame]["whole"]):
                probe_hits.append(f"{frame}: {probe}")
    print(f"\nhallucination probes fired (VLM side): {len(probe_hits)}")
    for h in probe_hits:
        print(f"  {h}")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
