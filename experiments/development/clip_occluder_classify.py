import json
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image

import clip
from recovery_pipeline import RecoveryPipeline

FRAMES = "live_disruption_frames"
OCC_DIR = "generated_occlusions"
OUT = "clip_occluder_classify.json"
SAMPLE_EVERY = 6
MAX_PER_CASE = 8

OCC_ANCHORS = [
    "a human hand", "a person's arm", "fingers covering an object",
    "a sheet of paper", "a book", "a cloth", "fabric covering something",
    "someone's sleeve",
]
REP_ANCHORS = [
    "a cardboard box", "a plastic bottle", "a coffee mug", "a metal can",
    "a computer mouse", "an object sitting on a desk", "an empty desk surface",
    "a bottle of water",
]

REAL_CASES = [
    ("occlusion_full", "disrupt", True),
    ("impostor", "recover", False),
    ("occlusion_remove", "recover", False),
]


def sample_frames(scenario, phase):
    out = []
    for fn in sorted(os.listdir(FRAMES)):
        m = re.match(rf"^{re.escape(scenario)}_{phase}_(\d+)\.jpg$", fn)
        if m:
            out.append(os.path.join(FRAMES, fn))
    return out[::SAMPLE_EVERY][:MAX_PER_CASE]


def crop_region(frame, box):
    x1, y1, x2, y2 = [max(0, v) for v in box]
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
    return frame[max(0, y1 - py):min(h, y2 + py),
                 max(0, x1 - px):min(w, x2 + px)]


def main():
    man = json.load(open(os.path.join(OCC_DIR, "manifest.json")))
    box = man["box"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load("ViT-B/32", device=device)
    cmodel.eval()

    with torch.no_grad():
        occ_tok = clip.tokenize(OCC_ANCHORS).to(device)
        rep_tok = clip.tokenize(REP_ANCHORS).to(device)
        occ_emb = cmodel.encode_text(occ_tok).float()
        rep_emb = cmodel.encode_text(rep_tok).float()
        occ_emb /= occ_emb.norm(dim=-1, keepdim=True)
        rep_emb /= rep_emb.norm(dim=-1, keepdim=True)

    def clip_text_verdict(word):
        with torch.no_grad():
            t = clip.tokenize([word]).to(device)
            e = cmodel.encode_text(t).float()
            e /= e.norm(dim=-1, keepdim=True)
            so = (e @ occ_emb.T).max().item()
            sr = (e @ rep_emb.T).max().item()
        return so > sr, so, sr

    def clip_image_verdict(crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            e = cmodel.encode_image(img).float()
            e /= e.norm(dim=-1, keepdim=True)
            so = (e @ occ_emb.T).max().item()
            sr = (e @ rep_emb.T).max().item()
        return so > sr, so, sr

    from run_study import make_fast_verifier
    verifier = make_fast_verifier()
    probe_v = getattr(verifier, "generate", None)
    if probe_v is None:
        raise SystemExit("ABORT: verifier has no .generate -- probe needs a "
                         "generative model.")

    pipe = RecoveryPipeline("red cup", tracker=None, verifier=None,
                            use_detector=False, use_opencv=True, use_vlm=False,
                            use_state=True, use_region_probe=True,
                            probe_verifier=probe_v)
    pipe.box = box

    items = [(it["file"], it["is_occluder"], "inpaint:" + it["name"])
             for it in man["items"]]
    for scen, phase, want in REAL_CASES:
        for p in sample_frames(scen, phase):
            items.append((p, want, f"real:{scen}/{phase}"))

    rows = []
    print(f"{'source':<26}{'truth':>9} | {'VLM word':>14} {'kw':>4} | "
          f"{'CLIPtxt':>8} {'CLIPimg':>8}")
    print("-" * 84)
    for path, truth, tag in items:
        frame = cv2.imread(path)
        if frame is None:
            continue
        label, kw_occ = pipe.probe_region(frame)
        crop = crop_region(frame, box)
        ct_occ, cto, ctr = clip_text_verdict(label or "")
        ci_occ, cio, cir = clip_image_verdict(crop)
        rows.append(dict(source=tag, truth_occluder=truth, vlm_word=label,
                         kw=bool(kw_occ), clip_text=bool(ct_occ),
                         clip_image=bool(ci_occ),
                         ct_so=round(cto, 3), ct_sr=round(ctr, 3),
                         ci_so=round(cio, 3), ci_sr=round(cir, 3)))
        mark = lambda v: ("OCC" if v else "rep")
        print(f"{tag:<26}{('OCC' if truth else 'rep'):>9} | {str(label):>14} "
              f"{mark(kw_occ):>4} | {mark(ct_occ):>8} {mark(ci_occ):>8}")

    def score(key):
        c = sum(1 for r in rows if bool(r[key]) == r["truth_occluder"])
        dang = sum(1 for r in rows
                   if r[key] and not r["truth_occluder"])
        safe = sum(1 for r in rows
                   if not r[key] and r["truth_occluder"])
        return c, len(rows), dang, safe

    print("\n" + "=" * 84)
    print(f"{'classifier':<14}{'correct':>10}{'dangerous(false-present)':>26}"
          f"{'safe(false-missing)':>22}")
    for key in ("kw", "clip_text", "clip_image"):
        c, n, dang, safe = score(key)
        print(f"{key:<14}{f'{c}/{n}':>10}{dang:>26}{safe:>22}")

    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
