import json
import os
import sys
import time

import cv2

import disruption_bench as db

RESULTS = "study_results.json"
VLM_MAX_PIXELS = 100352


def make_verifier(max_pixels=VLM_MAX_PIXELS, max_new_tokens=8):
    import torch
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from PIL import Image
    hf = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(hf, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        hf, dtype=torch.float16, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"))
    model.eval()

    def verify(prompt, crop_bgr, tokens=None):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, do_sample=False,
                                 max_new_tokens=tokens or max_new_tokens)
        return proc.decode(out[0][inp["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

    verify.model = model
    verify.proc = proc
    return verify


YES_WORDS = ("Yes", "yes", " Yes", " yes", "YES")
NO_WORDS = ("No", "no", " No", " no", "NO")


def make_fast_verifier(max_pixels=VLM_MAX_PIXELS, return_prob=False):
    import torch
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig)
    from PIL import Image
    hf = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(hf, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        hf, dtype=torch.float16, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"))
    model.eval()

    tok = proc.tokenizer

    def first_ids(words):
        out = set()
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                out.add(ids[0])
        return sorted(out)

    yes_ids = first_ids(YES_WORDS)
    no_ids = first_ids(NO_WORDS)
    if not yes_ids or not no_ids:
        raise RuntimeError("could not resolve yes/no token ids -- refusing to "
                           "guess, since a wrong id silently inverts every "
                           "verification the pipeline makes")

    def verify(prompt, crop_bgr):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inp).logits[0, -1].float()
        y = max(logits[i].item() for i in yes_ids)
        n = max(logits[i].item() for i in no_ids)
        ans = "yes" if y > n else "no"
        return (ans, y - n) if return_prob else ans

    def generate(prompt, crop_bgr, tokens=12):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=tokens, do_sample=False)
        return proc.decode(out[0][inp["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

    verify.generate = generate
    verify.model = model
    verify.proc = proc
    return verify


CONDITIONS = {
    "A_vlm_only":  dict(use_detector=False, use_opencv=False, use_vlm=True,
                        use_state=False),
    "B_detector":  dict(use_detector=True,  use_opencv=True,  use_vlm=False,
                        use_state=False),
    "C_stateless": dict(use_detector=True,  use_opencv=True,  use_vlm=True,
                        use_state=False),
    "D_full":      dict(use_detector=True,  use_opencv=True,  use_vlm=True,
                        use_state=True),
    "E_no_vlm":    dict(use_detector=True,  use_opencv=True,  use_vlm=False,
                        use_state=True),
    "F_probe":     dict(use_detector=True,  use_opencv=True,  use_vlm=True,
                        use_state=True, use_region_probe=True),
}


def score_episode(pipe, ep, frames):
    from recovery_pipeline import CONFIRMED
    pipe.reset()
    gt = ep["gt"]
    meta = ep["meta"]
    rec_start = meta["recover_start"]

    false_belief = 0
    lost_while_present = 0
    dis_correct = dis_n = 0
    stable_ok = stable_n = 0
    recovered_at = None

    for i, path in enumerate(frames):
        img = cv2.imread(path)
        if img is None:
            continue
        pipe.step(img)
        truth = gt[i]["present"]
        believes = pipe.believes_present()

        if truth is None:
            continue

        if i < meta["disrupt_start"]:
            stable_n += 1
            stable_ok += (believes == truth)

        if meta["disrupt_start"] <= i < rec_start:
            dis_n += 1
            dis_correct += (believes == truth)

        if believes and not truth:
            false_belief += 1
        if truth and not believes:
            lost_while_present += 1

        if i >= rec_start and recovered_at is None and believes == truth:
            recovered_at = i - rec_start

    ended_confirmed = pipe.status == CONFIRMED
    final_truth = next((g["present"] for g in reversed(gt)
                        if g.get("present") is not None), None)
    target_gone = final_truth is False
    return {
        "recovered": recovered_at is not None,
        "time_to_recovery": recovered_at,
        "false_belief_frames": false_belief,
        "lost_while_present": lost_while_present,
        "during_disruption_acc": round(dis_correct / dis_n, 3) if dis_n else None,
        "spurious_reacquisition": bool(ended_confirmed and target_gone),
        "stable_accuracy": round(stable_ok / stable_n, 3) if stable_n else None,
        "final_status": pipe.status,
        "vlm_calls": pipe.vlm_calls,
    }


def main():
    from recovery_pipeline import RecoveryPipeline
    from yolo_tracker import YoloTracker

    idx = db.load()
    targets = {e["meta"].get("target", "unknown") for e in idx.values()}
    if len(targets) != 1 or "unknown" in targets:
        raise SystemExit(f"ABORT: episodes have inconsistent/unknown target "
                         f"{targets} -- regenerate with disruption_bench.py")
    target = targets.pop()
    print(f"{len(idx)} episodes, {sum(e['n'] for e in idx.values())} frames, "
          f"target={target!r}")

    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print("loading verifier...")
    t0 = time.time()
    verifier = make_verifier()
    print(f"  verifier ready in {time.time() - t0:.0f}s")

    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    if want:
        unknown = [c for c in want if c not in CONDITIONS]
        if unknown:
            raise SystemExit(f"ABORT: unknown condition(s) {unknown}. "
                             f"Valid: {list(CONDITIONS)}")
        print(f"re-running ONLY: {want}  (others carried over)")

    all_res = {}
    if want and os.path.exists(RESULTS):
        prev = json.load(open(RESULTS))
        all_res = prev.get("per_episode", prev)
        stale = [c for c in want if c in all_res]
        print(f"  carrying over {[c for c in all_res if c not in want]}; "
              f"replacing {stale}")

    for cond, flags in CONDITIONS.items():
        if want and cond not in want:
            continue
        print(f"\n=== {cond} ===")
        pipe = RecoveryPipeline(target, tracker=tracker, verifier=verifier,
                                probe_verifier=verifier, **flags)
        per_ep = {}
        t_cond = time.time()
        for k, (name, ep) in enumerate(idx.items(), 1):
            t0 = time.time()
            r = score_episode(pipe, ep, ep["frames"])
            r["elapsed"] = round(time.time() - t0, 1)
            r["kind"] = ep["meta"]["kind"]
            r["type"] = ep["meta"]["type"]
            r["disrupt_frames"] = ep["meta"]["disrupt_frames"]
            per_ep[name] = r
            print(f"  [{k:>2}/{len(idx)}] {name:<28} "
                  f"durAcc={str(r['during_disruption_acc']):<6} "
                  f"lost={r['lost_while_present']:<3} "
                  f"false={r['false_belief_frames']:<3} "
                  f"spur={str(r['spurious_reacquisition']):<5} "
                  f"vlm={r['vlm_calls']}")
        print(f"  -- {cond} done in {time.time() - t_cond:.0f}s")
        all_res[cond] = per_ep
        json.dump(all_res, open(RESULTS, "w"), indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY -- C vs D isolates whether the reasoning STRUCTURE helps")
    print("=" * 78)
    hdr = (f"{'condition':<14}{'durAcc':>8}{'lost':>7}{'false':>7}"
           f"{'spur':>6}{'recov':>8}{'vlm':>7}")
    print(hdr); print("-" * len(hdr))
    for cond, eps in all_res.items():
        n = len(eps)
        rec = sum(e["recovered"] for e in eps.values())
        das = [e["during_disruption_acc"] for e in eps.values()
               if e["during_disruption_acc"] is not None]
        da = round(sum(das) / len(das), 3) if das else None
        lost = sum(e["lost_while_present"] for e in eps.values())
        fb = sum(e["false_belief_frames"] for e in eps.values())
        sp = sum(e["spurious_reacquisition"] for e in eps.values())
        vc = sum(e["vlm_calls"] for e in eps.values())
        print(f"{cond:<14}{str(da):>8}{lost:>7}{fb:>7}{sp:>6}"
              f"{rec}/{n:<6}{vc:>7}")
    print("\ndurAcc = belief correctness DURING the disruption (the real test)")
    print("lost   = frames the target was PRESENT but reported absent")
    print("false  = frames the target was ABSENT but reported present")

    kinds = sorted({e["meta"]["kind"] for e in idx.values()})
    print("\n" + "=" * 78)
    print("BY DISRUPTION KIND  (durAcc / lost / false)")
    print("=" * 78)
    hdr = f"{'condition':<14}" + "".join(f"{k:>16}" for k in kinds)
    print(hdr); print("-" * len(hdr))
    by_kind = {}
    for cond, eps in all_res.items():
        cells = []
        by_kind[cond] = {}
        for k in kinds:
            sel = [e for e in eps.values() if e["kind"] == k]
            das = [e["during_disruption_acc"] for e in sel
                   if e["during_disruption_acc"] is not None]
            da = round(sum(das) / len(das), 3) if das else None
            lost = sum(e["lost_while_present"] for e in sel)
            fb = sum(e["false_belief_frames"] for e in sel)
            by_kind[cond][k] = {"durAcc": da, "lost": lost, "false": fb,
                                "n": len(sel)}
            cells.append(f"{str(da):>7}/{lost:>3}/{fb:<3}")
        print(f"{cond:<14}" + "".join(f"{c:>16}" for c in cells))

    if "identity" in kinds and "D_full" in all_res and "E_no_vlm" in all_res:
        d = by_kind["D_full"]["identity"]
        e = by_kind["E_no_vlm"]["identity"]
        d_spur = sum(all_res["D_full"][n]["spurious_reacquisition"]
                     for n in all_res["D_full"]
                     if all_res["D_full"][n]["kind"] == "identity")
        e_spur = sum(all_res["E_no_vlm"][n]["spurious_reacquisition"]
                     for n in all_res["E_no_vlm"]
                     if all_res["E_no_vlm"][n]["kind"] == "identity")
        print("\n" + "=" * 78)
        print("D vs E ON SAME-CLASS IMPOSTORS -- does the VLM earn its cost?")
        print("=" * 78)
        print(f"  n = {d['n']} identity episodes")
        print(f"  D (with VLM)  false_belief={d['false']:<4} "
              f"spurious_reacq={d_spur}/{d['n']}")
        print(f"  E (no VLM)    false_belief={e['false']:<4} "
              f"spurious_reacq={e_spur}/{e['n']}")
        if e["false"] > d["false"]:
            print("  -> The VLM REJECTS impostors that geometry accepts. This "
                  "is the\n     capability the earlier D==E result could not "
                  "see, because no\n     episode required it.")
        elif d["false"] >= e["false"]:
            print("  -> The VLM did NOT reject the impostor. D==E now stands "
                  "as a real\n     negative result: verification was tested "
                  "and did not pay off.")

    print("\nPer-type recovery (condition D):")
    types = {}
    for name, r in all_res.get("D_full", {}).items():
        types.setdefault(r["type"], []).append(r)
    for t, rs in sorted(types.items()):
        rec = sum(r["recovered"] for r in rs)
        fb = sum(r["false_belief_frames"] for r in rs)
        lost = sum(r["lost_while_present"] for r in rs)
        print(f"  {t:<22} recovered={rec}/{len(rs)}  lost={lost:<3} "
              f"false_belief={fb}")
    json.dump({"per_episode": all_res, "by_kind": by_kind},
              open(RESULTS, "w"), indent=2)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
