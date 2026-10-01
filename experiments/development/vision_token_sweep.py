"""
vision_token_sweep.py -- make the ALREADY-VALIDATED model fast.

qwen2.5vl-3b on the transformers runtime measured 0/25 degenerate and held-out
acc 0.875 (vs 0.750 for the llama.cpp incumbent). Its ONLY failure is latency:
5.6s for an 8-token answer. That is not a generation cost -- 8 tokens is 8
forward passes. It is the VISION ENCODE.

Qwen2.5-VL uses DYNAMIC RESOLUTION: it tokenises the image into 28x28 patches
with no fixed cap, so a 1280x720 frame becomes THOUSANDS of vision tokens, and
every one of them is attended over on every generated token. For short
structured answers the vision encode is essentially the whole bill.

PROGRESS.md already establishes that 480x360 is sufficient for this project's
accuracy -- the FAST-XOR-ACCURATE ceiling was about llama.cpp's CUDA kernel,
not about the pixels being necessary. So capping max_pixels should buy a large
speedup at little accuracy cost.

This sweeps the cap and reports latency AND whether answers stay correct, so
the cap is chosen on evidence rather than assumed. Accuracy is checked on the
same held-out points as everything else.

Camera/arm untouched -- saved frames only.
"""
import json
import time

import cv2

from heldout_gt import HELDOUT_GT
import open_vocab_detect as ov

HF_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SHORT_TOKENS = 8
TRIALS = 6

# max_pixels caps, in pixels. Qwen packs 28x28 per visual token, so
# tokens ~= max_pixels / 784. Chosen to bracket the useful range:
#   200704 -> ~256 tokens  (~448x448)
#   401408 -> ~512 tokens  (~640x627)
#   802816 -> ~1024 tokens
#   None   -> uncapped, i.e. what the 5.6s measurement used
CAPS = [100352, 200704, 401408, 802816, None]


def build(max_pixels):
    import torch
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    kw = {}
    if max_pixels:
        kw["max_pixels"] = max_pixels
    proc = AutoProcessor.from_pretrained(HF_ID, **kw)
    model = AutoModelForImageTextToText.from_pretrained(
        HF_ID, dtype=torch.float16, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"))
    model.eval()
    return torch, proc, model


def ask(torch, proc, model, prompt, img_bgr, max_tokens):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n_vis = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=False)
    txt = proc.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
    return txt, n_vis


def main():
    frames = {f: cv2.imread(f) for f in HELDOUT_GT}
    frames = {f: im for f, im in frames.items() if im is not None}
    if not frames:
        raise SystemExit("no held-out frames readable")
    probe = list(frames.values())[0]

    results = {}
    for cap in CAPS:
        label = str(cap) if cap else "uncapped"
        print("\n" + "=" * 66)
        print(f"max_pixels = {label}")
        print("=" * 66)
        try:
            torch, proc, model = build(cap)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            results[label] = {"error": str(e)}
            continue

        # latency at the pipeline's real budget
        lat, ntok = [], 0
        for i in range(TRIALS):
            jit = cv2.convertScaleAbs(probe, alpha=1.0 + (i % 3) * 0.02)
            t0 = time.time()
            _, ntok = ask(torch, proc, model,
                          "Answer in one word: is a keyboard visible?",
                          jit, SHORT_TOKENS)
            lat.append(time.time() - t0)
        lat.sort()
        p50 = round(lat[len(lat) // 2], 3)

        # accuracy on the SAME held-out points as every other number
        tp = fp = tn = fn = 0
        for f, im in frames.items():
            txt, _ = ask(torch, proc, model, ov.LIST_PROMPT, im, ov.MAX_TOKENS)
            items = ov._dedupe(ov._parse_items(txt)) if txt else []
            for obj, gt in HELDOUT_GT[f].items():
                pred = ov.is_present(obj, items)
                tp += gt and pred; fn += gt and not pred
                fp += (not gt) and pred; tn += (not gt) and not pred
        acc = round((tp + tn) / (tp + fp + tn + fn), 3)

        results[label] = {"latency_short_p50": p50, "prompt_tokens": ntok,
                          "acc": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        print(f"  short p50 {p50}s | prompt tokens {ntok} | held-out acc {acc}")

        del model, proc
        import gc; gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        json.dump(results, open("vision_token_sweep.json", "w"), indent=2)

    print("\n" + "=" * 66)
    print("SUMMARY   target: short p50 < 1.0s, acc >= 0.750")
    print("=" * 66)
    print(f"{'max_pixels':<12}{'short p50':>11}{'tokens':>9}{'acc':>7}")
    print("-" * 39)
    for k, v in results.items():
        if "error" in v:
            print(f"{k:<12}  {v['error'][:40]}")
        else:
            print(f"{k:<12}{v['latency_short_p50']:>11}"
                  f"{v['prompt_tokens']:>9}{v['acc']:>7}")


if __name__ == "__main__":
    main()
