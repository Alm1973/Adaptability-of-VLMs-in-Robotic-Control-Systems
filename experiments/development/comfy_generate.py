import json
import os
import sys
import time
import urllib.parse
import urllib.request

HOST = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
NEG = "blurry, low quality, distorted, watermark, text"


def workflow(prompt, seed, width=512, height=512, steps=20, cfg=7.0):
    return {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0, "model": ["4", 0],
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "avi_gen", "images": ["8", 0]}},
    }


def post(path, payload):
    req = urllib.request.Request(
        f"{HOST}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path):
    return json.loads(urllib.request.urlopen(f"{HOST}{path}",
                                             timeout=60).read())


def await_image(pid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get(f"/history/{pid}")
        if pid in hist:
            imgs = [i for o in hist[pid]["outputs"].values()
                    for i in o.get("images", [])]
            if not imgs:
                raise SystemExit(f"ABORT: {pid} finished with no image")
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            return urllib.request.urlopen(f"{HOST}/view?{q}",
                                          timeout=180).read()
        time.sleep(1.5)
    raise SystemExit(f"ABORT: {pid} timed out after {timeout}s")


def generate(prompt, out_path, seed=None, timeout=600):
    seed = int(time.time()) % 2**31 if seed is None else int(seed)
    wf = workflow(prompt, seed)
    pid = post("/prompt", {"prompt": wf})["prompt_id"]
    print(f"queued {pid}  seed={seed}")

    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get(f"/history/{pid}")
        if pid in hist:
            outs = hist[pid]["outputs"]
            imgs = [i for o in outs.values() for i in o.get("images", [])]
            if not imgs:
                raise SystemExit("ABORT: finished with no image")
            im = imgs[0]
            q = urllib.parse.urlencode({"filename": im["filename"],
                                        "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            data = urllib.request.urlopen(f"{HOST}/view?{q}",
                                          timeout=120).read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"wrote {out_path}  {len(data)} bytes  "
                  f"{time.time() - t0:.1f}s  seed={seed}")
            return seed
        time.sleep(1.0)
    raise SystemExit(f"ABORT: timed out after {timeout}s")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    generate(sys.argv[1], sys.argv[2],
             sys.argv[3] if len(sys.argv) > 3 else None)
