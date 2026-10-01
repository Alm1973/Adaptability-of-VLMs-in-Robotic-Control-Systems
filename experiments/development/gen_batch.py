import json
import os
import time

import comfy_generate as cg

OUTDIR = "generated_frames"

JOBS = [
    ("present_01", 1002, "a red disposable plastic cup on a cluttered office "
                         "desk, photograph, natural lighting"),
    ("present_02", 1003, "a red plastic cup next to a black keyboard on a "
                         "white desk, photograph"),
    ("present_03", 1004, "a red plastic cup on a wooden table, daylight, "
                         "photograph"),
    ("absent_00", 2001, "an empty white office desk with a black keyboard and "
                        "a mouse, photograph, natural lighting"),
    ("absent_01", 2002, "a wooden table with an open book and a pen, "
                        "photograph, daylight"),
    ("absent_02", 2003, "an office desk with a laptop and a notebook, "
                        "photograph"),
    ("absent_03", 2004, "a bare desk surface with a green water bottle, "
                        "photograph"),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    queued, done = [], []
    for name, seed, prompt in JOBS:
        out = os.path.join(OUTDIR, f"{name}.png")
        if os.path.exists(out):
            print(f"skip {name} (exists)")
            continue
        wf = cg.workflow(prompt, seed, width=512, height=512, steps=20)
        pid = cg.post("/prompt", {"prompt": wf})["prompt_id"]
        queued.append((name, seed, prompt, pid, out))
        print(f"queued {name}  seed={seed}  {pid}")

    t0 = time.time()
    for name, seed, prompt, pid, out in queued:
        while True:
            h = cg.get(f"/history/{pid}")
            if pid in h:
                imgs = [i for o in h[pid]["outputs"].values()
                        for i in o.get("images", [])]
                if imgs:
                    import urllib.parse
                    import urllib.request
                    im = imgs[0]
                    q = urllib.parse.urlencode(
                        {"filename": im["filename"],
                         "subfolder": im.get("subfolder", ""),
                         "type": im.get("type", "output")})
                    data = urllib.request.urlopen(
                        f"{cg.HOST}/view?{q}", timeout=180).read()
                    open(out, "wb").write(data)
                    print(f"  wrote {out}  ({time.time() - t0:.0f}s elapsed)")
                    done.append({"name": name, "seed": seed,
                                 "prompt": prompt, "file": out})
                break
            time.sleep(3)

    json.dump(done, open("generated_frames/manifest.json", "w"), indent=2)
    print(f"\n{len(done)} images in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
