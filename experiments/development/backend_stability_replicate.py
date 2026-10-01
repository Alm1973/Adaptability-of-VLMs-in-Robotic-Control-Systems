import glob
import json

import cv2
import numpy as np

from backend_latency_ab import (WH, PROMPT, encode_b64, call, kill_all,
                                launch_ollama, launch_b10326, is_degenerate,
                                health_ok)

R = 3
N = 15


def unique_images(seed, n):
    rng = np.random.default_rng(seed)
    bases = [cv2.imread(f) for f in sorted(glob.glob("exp_*.jpg"))]
    out = []
    for i in range(n):
        base = bases[i % len(bases)]
        h, w = base.shape[:2]
        cw, ch = int(w * 0.9), int(h * 0.9)
        x = int(rng.integers(0, w - cw + 1))
        y = int(rng.integers(0, h - ch + 1))
        img = cv2.resize(base[y:y + ch, x:x + cw], WH)
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=int(rng.integers(-25, 26)))
        out.append(encode_b64(img))
    return out


def one_run(name, launcher, port, b64s):
    kill_all()
    p = launcher()
    if p is None:
        return {"error": "launch failed"}
    first_lock, degen, seq = None, 0, []
    for i, b64 in enumerate(b64s):
        try:
            out, _ = call(port, b64)
        except Exception as e:
            seq.append("E")
            continue
        d = is_degenerate(out)
        seq.append("D" if d else ".")
        if d:
            degen += 1
            if first_lock is None:
                first_lock = i + 1
    kill_all()
    return {"first_lock": first_lock, "degen": degen, "n": len(seq),
            "seq": "".join(seq)}


def main():
    results = {"ollama": [], "b10326": []}
    for run in range(R):
        b64s = unique_images(seed=1000 + run, n=N)
        print(f"\n########## RUN {run+1}/{R} (seed {1000+run}) ##########")
        a = one_run("ollama", launch_ollama, 18080, b64s)
        print(f"  A ollama : {a}")
        b = one_run("b10326", launch_b10326, 18081, b64s)
        print(f"  B b10326 : {b}")
        results["ollama"].append(a)
        results["b10326"].append(b)

    print("\n\n============ STABILITY REPLICATION (480x360, unique imgs) ============")
    print(f"R={R} runs x N={N} unique calls, fresh server each, GPU.\n")
    print(f"{'backend':<10} {'run':<5} {'first_lock':<12} {'degen/N':<9} seq")
    for name in ("ollama", "b10326"):
        for i, r in enumerate(results[name]):
            fl = r.get("first_lock")
            print(f"{name:<10} {i+1:<5} {str(fl):<12} "
                  f"{str(r.get('degen'))+'/'+str(r.get('n')):<9} {r.get('seq','')}")
    def summarize(rs):
        locks = [r["first_lock"] for r in rs if r.get("first_lock")]
        degens = [r["degen"] for r in rs if "degen" in r]
        clean_runs = sum(1 for r in rs if r.get("degen") == 0)
        return (f"clean_runs {clean_runs}/{len(rs)}  "
                f"first_locks {locks}  total_degen {sum(degens)}/"
                f"{sum(r.get('n',0) for r in rs)}")
    print("\nA ollama-bundled:", summarize(results["ollama"]))
    print("B b10326        :", summarize(results["b10326"]))
    json.dump(results, open("backend_stability_replicate.json", "w"), indent=2)
    print("\nwrote backend_stability_replicate.json")


if __name__ == "__main__":
    main()
