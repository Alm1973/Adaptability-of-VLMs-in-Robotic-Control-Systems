import gc
import json
import os
import re
import time

import cv2
import numpy as np

from heldout_gt import HELDOUT_GT, HALLUCINATION_PROBES
import open_vocab_detect as ov

SOAK_CALLS = 25
RESULTS = "vlm_bakeoff_results.json"

SHORT_TOKENS = 8
LONG_TOKENS = 60

CANDIDATES = [
    ("smolvlm2-2.2b", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "fp16",
     "built for speed; ~4.4GB fp16 FITS 6GB, so no dequant overhead at all"),
    ("qwen2.5vl-3b-CONTROL", "Qwen/Qwen2.5-VL-3B-Instruct", "4bit",
     "SAME weights as the incumbent, new runtime -- isolates runtime vs model"),
    ("qwen3vl-4b", "Qwen/Qwen3-VL-4B-Instruct", "4bit",
     "leading 2026 small VLM -- accuracy ceiling check; may not fit"),
    ("gemma3-4b", "google/gemma-3-4b-it", "4bit",
     "different lab -- hedge against a Qwen-family quirk"),
]

OCCLUSION_PROMPT = (
    "Look at the region described. Answer with exactly one word: "
    "VISIBLE if the object is fully visible, PARTIAL if something is in front "
    "of part of it, or HIDDEN if it is completely blocked. "
    "Is the {obj} VISIBLE, PARTIAL, or HIDDEN?"
)


def is_degenerate(text):
    if not text:
        return False
    return bool(re.search(r"(.)\1{7,}", text))


def unique_image(base, i):
    h, w = base.shape[:2]
    dx, dy = (i * 7) % 40, (i * 11) % 40
    crop = base[dy:h - 40 + dy, dx:w - 40 + dx]
    return cv2.convertScaleAbs(crop, alpha=1.0 + ((i % 5) - 2) * 0.04,
                               beta=((i % 7) - 3) * 3)


class HFBackend:

    def __init__(self, hf_id, mode="4bit"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(hf_id)
        quant = None
        if mode == "4bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch.float16, device_map="auto",
            quantization_config=quant)
        self.model.eval()

    def chat(self, prompt, image_bgr, max_tokens=140):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def free(self):
        del self.model, self.processor
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def soak(backend, base):
    degen = first_lock = 0
    lat_short, lat_long = [], []
    for i in range(SOAK_CALLS):
        img = unique_image(base, i)
        short = (i % 2 == 0)
        prompt = ("Answer in one word: is a keyboard visible?" if short
                  else "List the objects you see.")
        budget = SHORT_TOKENS if short else LONG_TOKENS
        t0 = time.time()
        try:
            out = backend.chat(prompt, img, max_tokens=budget)
        except Exception as e:
            print(f"    call {i}: EXCEPTION {type(e).__name__}: {e}")
            out = ""
        dt = time.time() - t0
        (lat_short if short else lat_long).append(dt)
        if is_degenerate(out):
            degen += 1
            if not first_lock:
                first_lock = i + 1
    lat_short.sort(); lat_long.sort()
    p50 = lambda v: round(v[len(v) // 2], 3) if v else None
    return {"degenerate": degen, "of": SOAK_CALLS,
            "first_lock_at": first_lock or None,
            "latency_short_p50": p50(lat_short),
            "latency_short_max": round(lat_short[-1], 3) if lat_short else None,
            "latency_long_p50": p50(lat_long)}


def heldout(backend):
    tp = fp = tn = fn = 0
    halluc = 0
    for frame, labels in HELDOUT_GT.items():
        img = cv2.imread(frame)
        if img is None:
            continue
        txt = backend.chat(ov.LIST_PROMPT, img, max_tokens=ov.MAX_TOKENS)
        items = ov._dedupe(ov._parse_items(txt)) if txt else []
        for obj, gt in labels.items():
            pred = ov.is_present(obj, items)
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1
        for probe in HALLUCINATION_PROBES:
            if ov.is_present(probe, items):
                halluc += 1
    p, n = tp + fn, tn + fp
    return {"acc": round((tp + tn) / (p + n), 3) if p + n else None,
            "sens": round(tp / p, 3) if p else None,
            "spec": round(tn / n, 3) if n else None,
            "hallucinations": halluc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def occlusion(backend):
    frame = "c2_10_home.jpg"
    img = cv2.imread(frame)
    if img is None:
        return {"error": "frame unreadable"}
    h, w = img.shape[:2]
    box = (int(w * 0.55), int(h * 0.60), int(w * 0.95), int(h * 0.98))
    cases = [("VISIBLE", 0.0), ("PARTIAL", 0.5), ("HIDDEN", 1.0)]
    correct = 0
    detail = []
    for expect, frac in cases:
        test = img.copy()
        if frac > 0:
            x1, y1, x2, y2 = box
            cover = int((y2 - y1) * frac)
            cv2.rectangle(test, (x1, y2 - cover), (x2, y2), (20, 20, 20), -1)
        out = backend.chat(OCCLUSION_PROMPT.format(obj="keyboard"), test,
                           max_tokens=8)
        got = (out or "").strip().upper()
        ok = expect in got
        correct += ok
        detail.append({"expected": expect, "got": out, "ok": ok})
    return {"correct": correct, "of": len(cases), "detail": detail}


def main():
    base = cv2.imread("c2_05_pan_b108_t90.jpg")
    if base is None:
        raise SystemExit("need c2_05_pan_b108_t90.jpg for the soak")

    import sys
    only = [a.lower() for a in sys.argv[1:]]
    all_results = {}
    if os.path.exists(RESULTS):
        try:
            all_results = json.load(open(RESULTS))
        except Exception:
            pass

    for label, hf_id, mode, why in CANDIDATES:
        if only and not any(o in label.lower() for o in only):
            continue
        print("\n" + "=" * 72)
        print(f"{label}   ({hf_id})  [{mode}]")
        print(f"  {why}")
        print("=" * 72)
        try:
            t0 = time.time()
            be = HFBackend(hf_id, mode)
            print(f"  loaded in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            all_results[label] = {"load_error": f"{type(e).__name__}: {e}"}
            json.dump(all_results, open(RESULTS, "w"), indent=2)
            continue

        r = {"hf_id": hf_id, "why": why}
        try:
            print("  [1/3] unique-image soak...")
            r["soak"] = soak(be, base)
            print(f"        {r['soak']}")
            print("  [2/3] held-out accuracy...")
            r["heldout"] = heldout(be)
            print(f"        {r['heldout']}")
            print("  [3/3] occlusion reasoning...")
            r["occlusion"] = occlusion(be)
            print(f"        {r['occlusion']['correct']}/{r['occlusion']['of']}")
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"  ERROR: {r['error']}")
        finally:
            be.free()

        all_results[label] = r
        json.dump(all_results, open(RESULTS, "w"), indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY   (incumbent reference: acc 0.750, ~3.2s, LOCKS at call 3-4)")
    print("  SHORT s = latency at the pipeline's real token budget <- HEADLINE")
    print("  target: degen 0, SHORT < 1.0s, acc >= 0.750")
    print("=" * 72)
    hdr = (f"{'model':<24}{'degen':>8}{'SHORT s':>9}{'long s':>8}"
           f"{'acc':>7}{'halluc':>8}{'occl':>7}")
    print(hdr); print("-" * len(hdr))
    for label, r in all_results.items():
        if "soak" not in r:
            print(f"{label:<24}  {r.get('load_error', r.get('error', '?'))[:40]}")
            continue
        s, h, o = r["soak"], r.get("heldout", {}), r.get("occlusion", {})
        print(f"{label:<24}{s['degenerate']:>4}/{s['of']:<3}"
              f"{str(s.get('latency_short_p50')):>9}"
              f"{str(s.get('latency_long_p50')):>8}"
              f"{str(h.get('acc')):>7}{h.get('hallucinations', '?'):>8}"
              f"  {o.get('correct', '?')}/{o.get('of', '?')}")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
