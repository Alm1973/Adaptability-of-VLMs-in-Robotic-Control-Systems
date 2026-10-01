import json
import os

import cv2
import numpy as np

import comfy_generate as cg
from inpaint_occluders import BASE, CROP, upload, workflow

OUTDIR = "generated_corpus"

ITEMS = [
    ("occ_hand", "a human hand covering the object, skin, fingers, "
                 "photograph", True),
    ("occ_forearm", "a bare human forearm across the scene, skin, "
                    "photograph", True),
    ("occ_book", "a closed hardcover book standing upright, photograph", True),
    ("occ_newspaper", "a folded newspaper standing upright, photograph", True),
    ("occ_paper", "a plain white sheet of paper, photograph", True),
    ("occ_towel", "a grey folded towel, fabric, photograph", True),
    ("occ_glove", "a black work glove, photograph", True),
    ("occ_folder", "a manila file folder standing upright, photograph", True),
    ("occ_notebook", "a spiral notebook standing upright, photograph", True),
    ("occ_box", "a brown cardboard box, photograph", True),

    ("rep_mug", "a white ceramic coffee mug on the desk, photograph", False),
    ("rep_bottle", "a clear plastic water bottle standing on the desk, "
                   "photograph", False),
    ("rep_plant", "a small potted green plant on the desk, photograph", False),
    ("rep_apple", "a red apple on the desk, photograph", False),
    ("rep_canister", "a metal thermos flask on the desk, photograph", False),
    ("rep_speaker", "a small black bluetooth speaker on the desk, "
                    "photograph", False),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    frame = cv2.imread(BASE)
    if frame is None:
        raise SystemExit(f"ABORT: no {BASE}")
    H, W = frame.shape[:2]

    prev = json.load(open("generated_occlusions/manifest.json"))
    x1, y1, x2, y2 = prev["box"]
    ox, oy = prev["items"][0]["crop_origin"]
    crop = frame[oy:oy + CROP, ox:ox + CROP].copy()

    pad = 26
    mask = np.zeros((CROP, CROP), np.uint8)
    mask[max(0, y1 - oy - pad):min(CROP, y2 - oy + pad),
         max(0, x1 - ox - pad):min(CROP, x2 - ox + pad)] = 255

    img_name = upload(crop, "avi_base_crop.png")
    mask_name = upload(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                       "avi_base_mask.png")
    print(f"base {BASE}  box {x1},{y1},{x2},{y2}  crop {ox},{oy}")

    manifest, pending = [], []
    for i, (name, prompt, is_occ) in enumerate(ITEMS):
        out = os.path.join(OUTDIR, f"{name}.png")
        if os.path.exists(out):
            print(f"skip {name}")
            manifest.append({"name": name, "prompt": prompt,
                             "is_occluder": is_occ, "file": out})
            continue
        wf = workflow(img_name, mask_name, prompt, 4000 + i)
        pid = cg.post("/prompt", {"prompt": wf})["prompt_id"]
        pending.append((name, prompt, is_occ, pid, out))
        print(f"queued {name} ({pid[:8]})")

    for name, prompt, is_occ, pid, out in pending:
        data = cg.await_image(pid)
        patch = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        full = frame.copy()
        full[oy:oy + CROP, ox:ox + CROP] = patch
        cv2.imwrite(out, full)
        manifest.append({"name": name, "prompt": prompt,
                         "is_occluder": is_occ, "file": out})
        print(f"  wrote {out}")

    json.dump({"base": BASE, "box": [x1, y1, x2, y2], "items": manifest},
              open(f"{OUTDIR}/manifest.json", "w"), indent=2)
    print(f"\n{len(manifest)} items in {OUTDIR}/")


if __name__ == "__main__":
    main()
