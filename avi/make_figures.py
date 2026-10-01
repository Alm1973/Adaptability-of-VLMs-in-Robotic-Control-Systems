import json
import os
import sys

import cv2

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-09-01_233533"
OUT = "figures"
COLS = 8
TILE_W = 240
VID_W = 640
VID_FPS = 10

KEY_TRIALS = {
    14: "removal_high_vlm_prevents_ghost",
    9: "identity_high_vlm_prevents_ghost",
    46: "removal_high_both_fail",
    31: "substitution_high_vlm_never_asked",
}
SCENARIO_PICK = {
    "control": 8, "occlude_hand": 10, "occlude_object": 27,
    "removal": 14, "substitution": 31, "identity_swap": 9,
    "lighting": 7, "camera_pose": 1,
}


def load(run):
    rows = json.load(open(os.path.join(run, "results.json")))
    return {r["trial_no"]: r for r in rows}


def label(img, text, sub=""):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text, (6, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (170, 210, 255), 1, cv2.LINE_AA)
    return img


def contact_sheet(run, row, n=16):
    frames = row.get("frames") or []
    if not frames:
        return None
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    tiles = []
    for i in idx:
        f = frames[i]
        p = os.path.join(run, f["annotated"])
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (TILE_W, round(h * TILE_W / w)))
        gt = "PRESENT" if f["present"] else "ABSENT"
        edge = " edge" if f["edge"] else ""
        tiles.append(label(img, f"{f['phase']} {f['t']:.1f}s{edge}",
                           f"{f['status']} | truth {gt}"))
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows_n = (len(tiles) + COLS - 1) // COLS
    sheet = cv2.copyMakeBorder(
        cv2.resize(tiles[0], (tw, th)), 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    canvas = 255 * \
        __import__("numpy").ones((rows_n * th, COLS * tw, 3), dtype="uint8")
    canvas[:] = 18
    for k, t in enumerate(tiles):
        r, c = divmod(k, COLS)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return canvas


def video(run, row, path):
    frames = row.get("frames") or []
    if not frames:
        return False
    first = cv2.imread(os.path.join(run, frames[0]["annotated"]))
    if first is None:
        return False
    h, w = first.shape[:2]
    size = (VID_W, round(h * VID_W / w))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VID_FPS, size)
    if not vw.isOpened():
        return False
    for f in frames:
        img = cv2.imread(os.path.join(run, f["annotated"]))
        if img is None:
            continue
        vw.write(cv2.resize(img, size))
    vw.release()
    return True


def main():
    import numpy  # noqa: F401  (used inside contact_sheet)
    trials = load(RUN)
    os.makedirs(os.path.join(OUT, "contact_sheets"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)

    bdir = os.path.join(RUN, "baseline")
    bimgs = sorted(x for x in os.listdir(bdir) if x.endswith(".jpg"))
    if bimgs:
        tiles = []
        for fn in bimgs[:8]:
            img = cv2.imread(os.path.join(bdir, fn))
            if img is None:
                continue
            h, w = img.shape[:2]
            tiles.append(label(cv2.resize(img, (TILE_W, round(h * TILE_W / w))),
                               fn.split("_")[1], "baseline scan"))
        if tiles:
            th, tw = tiles[0].shape[:2]
            canvas = numpy.zeros((th, len(tiles) * tw, 3), dtype="uint8")
            for k, t in enumerate(tiles):
                canvas[:, k * tw:(k + 1) * tw] = t
            cv2.imwrite(os.path.join(OUT, "baseline_scan.jpg"), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            print("baseline_scan.jpg")

    for no, row in sorted(trials.items()):
        sheet = contact_sheet(RUN, row)
        if sheet is None:
            continue
        name = (f"trial_{no:03d}_{row['scenario']}_rep{row['rep']}"
                f"_{row['clutter']}.jpg")
        cv2.imwrite(os.path.join(OUT, "contact_sheets", name), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 78])
    print(f"{len(trials)} contact sheets")

    made = []
    for scen, no in SCENARIO_PICK.items():
        row = trials.get(no)
        if not row:
            continue
        p = os.path.join(OUT, "videos", f"{scen}_trial{no:03d}.mp4")
        if video(RUN, row, p):
            made.append(p)
    for no, tag in KEY_TRIALS.items():
        row = trials.get(no)
        if not row:
            continue
        p = os.path.join(OUT, "videos", f"KEY_{tag}_trial{no:03d}.mp4")
        if video(RUN, row, p):
            made.append(p)
    print(f"{len(made)} videos")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"figures/ total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
