import base64
import glob
import json
import statistics as st
import time
import urllib.request

import cv2

from openvocab_baseline import PROMPT, parse

PORT = 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
RESOLUTION = (480, 360)

NEGATIVE_CONTROLS = ["banana", "elephant", "bicycle", "umbrella"]

VERIFIED_POSITIVES = {
    "c2_07_tilt_b90_t78.jpg": {"laptop": 1, "water bottle": 1,
                               "keyboard": 0, "computer mouse": 0},
    "c2_10_home.jpg":         {"laptop": 1, "water bottle": 1,
                               "keyboard": 1, "computer mouse": 1},
}


def is_degenerate(text):
    if not text:
        return True
    s = text.strip()
    if len(s) < 5:
        return False
    mc = max(set(s), key=s.count)
    return s.count(mc) / len(s) > 0.8


def encode(path):
    img = cv2.imread(path)
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, RESOLUTION))
    return base64.b64encode(buf.tobytes()).decode()


def call(b64, prompt):
    payload = {
        "model": "qwen2.5vl", "max_tokens": 20, "cache_prompt": False,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"].strip(), time.time() - t0


def main():
    import llm_backend
    if not llm_backend.ensure_running():
        print("server not healthy; aborting")
        return
    frames = sorted(glob.glob("c2_*.jpg"))
    if not frames:
        print("no c2_*.jpg held-out frames found")
        return
    enc = {f: encode(f) for f in frames}
    print(f"HELD-OUT VALIDATION on {len(frames)} c2_* frames "
          f"(never used for tuning)\n")

    print("=== NEGATIVE CONTROLS (any 'yes' is a hallucination) ===")
    fp = tot = deg = 0
    lat = []
    fp_detail = []
    for f in frames:
        hits = []
        for obj in NEGATIVE_CONTROLS:
            try:
                out, dt = call(enc[f], PROMPT.format(obj=obj))
            except Exception as e:
                print(f"  ERROR {e}")
                out, dt = "", 0.0
            lat.append(dt)
            if is_degenerate(out):
                deg += 1
            tot += 1
            if parse(out) == 1:
                fp += 1
                hits.append(obj)
        if hits:
            fp_detail.append((f, hits))
        print(f"  {f:<26} FP: {hits if hits else 'none'}")
    fp_rate = fp / tot if tot else None

    print("\n=== VERIFIED POSITIVES (labeled by eye this session) ===")
    correct = ptot = 0
    misses = []
    for f, labels in VERIFIED_POSITIVES.items():
        if f not in enc:
            continue
        for obj, gt in labels.items():
            try:
                out, dt = call(enc[f], PROMPT.format(obj=obj))
            except Exception as e:
                print(f"  ERROR {e}")
                out = ""
            pred = parse(out)
            ptot += 1
            ok = (pred == gt)
            correct += ok
            if not ok:
                misses.append((f, obj, gt, pred))
            print(f"  {f:<26} {obj:<14} gt={gt} pred={pred} "
                  f"{'OK' if ok else 'XX'}")

    res = {
        "n_frames": len(frames),
        "neg_fp": fp, "neg_total": tot,
        "neg_fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "neg_fp_detail": fp_detail,
        "degenerate": deg,
        "pos_correct": correct, "pos_total": ptot, "pos_misses": misses,
        "med_lat_s": round(st.median(lat), 2) if lat else None,
    }
    json.dump(res, open("heldout_validation.json", "w"), indent=2)

    print("\n===== HELD-OUT SUMMARY =====")
    print(f"negative-control FP rate : {res['neg_fp_rate']} ({fp}/{tot})")
    print(f"verified-positive acc    : "
          f"{correct}/{ptot}" + (f"  ({correct/ptot:.2f})" if ptot else ""))
    print(f"degenerate outputs       : {deg}/{tot}   (must be 0)")
    print(f"median latency           : {res['med_lat_s']}s")
    print("\nIn-sample reference (exp_ frames): FP rate 0.343, macro 0.701")
    if fp_detail:
        print("\nhallucinated objects by frame:")
        for f, hits in fp_detail:
            print(f"  {f}: {hits}")


if __name__ == "__main__":
    main()
