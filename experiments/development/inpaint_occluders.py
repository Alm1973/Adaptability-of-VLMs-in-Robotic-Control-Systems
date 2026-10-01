import io
import json
import os
import urllib.request
import uuid

import cv2
import numpy as np

import comfy_generate as cg

BASE = "live_disruption_frames/occlusion_full_baseline_030.jpg"
OUTDIR = "generated_occlusions"
CKPT = cg.CKPT
CROP = 512

OCCLUDERS = [
    ("hand", "a human hand covering the object, skin, fingers, photograph",
     True),
    ("book", "a closed hardcover book standing upright, photograph", True),
    ("paper", "a plain white sheet of paper, photograph", True),
    ("sleeve", "a blue fabric sleeve of a jacket, cloth, photograph", True),
    ("box", "a brown cardboard box, photograph", True),
]


def upload(img_bgr, name):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise SystemExit("encode failed")
    body, bnd = b"", uuid.uuid4().hex
    for k, v in (("type", "input"), ("overwrite", "true")):
        body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\""
                 f"\r\n\r\n{v}\r\n").encode()
    body += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; "
             f"filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += buf.tobytes() + f"\r\n--{bnd}--\r\n".encode()
    req = urllib.request.Request(
        f"{cg.HOST}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["name"]


def workflow(img_name, mask_name, prompt, seed, steps=20, denoise=1.0):
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT}},
        "10": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry, low quality, distorted, watermark",
                         "clip": ["4", 1]}},
        "12": {"class_type": "VAEEncodeForInpaint",
               "inputs": {"pixels": ["10", 0], "vae": ["4", 2],
                          "mask": ["11", 0], "grow_mask_by": 12}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": denoise, "model": ["4", 0],
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["12", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "inpaint", "images": ["8", 0]}},
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    frame = cv2.imread(BASE)
    if frame is None:
        raise SystemExit(f"ABORT: no {BASE}")
    H, W = frame.shape[:2]

    from yolo_tracker import YoloTracker
    t = YoloTracker(default_target="red cup")
    t.set_targets(["red cup"], allow_unreliable=True, quiet=True)
    res = t.model.predict(frame, conf=0.1, verbose=False)[0]
    box = None
    for b in res.boxes:
        if t.model.names[int(b.cls)] == "red cup":
            c = float(b.conf)
            if box is None or c > box[0]:
                box = (c, [int(v) for v in b.xyxy[0]])
    if box is None:
        raise SystemExit("ABORT: no cup in the base frame")
    conf, (x1, y1, x2, y2) = box
    print(f"target box {x1},{y1},{x2},{y2}  conf {conf:.3f}")

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ox = max(0, min(W - CROP, cx - CROP // 2))
    oy = max(0, min(H - CROP, cy - CROP // 2))
    crop = frame[oy:oy + CROP, ox:ox + CROP].copy()

    pad = 26
    mask = np.zeros((CROP, CROP), np.uint8)
    mx1, my1 = max(0, x1 - ox - pad), max(0, y1 - oy - pad)
    mx2, my2 = min(CROP, x2 - ox + pad), min(CROP, y2 - oy + pad)
    mask[my1:my2, mx1:mx2] = 255
    print(f"crop origin {ox},{oy}  mask {mx1},{my1},{mx2},{my2}")

    img_name = upload(crop, "avi_base_crop.png")
    mask_name = upload(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                       "avi_base_mask.png")
    print(f"uploaded {img_name}, {mask_name}")

    manifest = []
    for i, (name, prompt, is_occ) in enumerate(OCCLUDERS):
        seed = 3000 + i
        wf = workflow(img_name, mask_name, prompt, seed)
        pid = cg.post("/prompt", {"prompt": wf})["prompt_id"]
        print(f"queued {name} ({pid[:8]})")
        data = cg.await_image(pid)
        patch = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        out = frame.copy()
        out[oy:oy + CROP, ox:ox + CROP] = patch
        path = os.path.join(OUTDIR, f"occ_{name}.png")
        cv2.imwrite(path, out)
        manifest.append({"name": name, "prompt": prompt, "seed": seed,
                         "is_occluder": is_occ, "file": path,
                         "box": [x1, y1, x2, y2],
                         "crop_origin": [ox, oy]})
        print(f"  wrote {path}")

    json.dump({"base": BASE, "box": [x1, y1, x2, y2],
               "items": manifest}, open(f"{OUTDIR}/manifest.json", "w"),
              indent=2)
    print(f"\n{len(manifest)} occluder variants written to {OUTDIR}/")


if __name__ == "__main__":
    main()
