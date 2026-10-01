import gc
import json
import time

import disruption_bench as db
from verifier_speed import build_crops, PROMPT

BUDGETS = [100352, 200704, 301056, 401408, 602112]
OUT = "pixel_budget.json"
WARMUP = 3


def looks_degenerate(ans):
    a = (ans or "").strip().lower()
    if not a:
        return True
    if a.startswith("y") or a.startswith("n"):
        return False
    stripped = a.replace(" ", "")
    return len(set(stripped)) <= 2 and len(stripped) >= 4


def main():
    import torch
    from run_study import make_verifier

    idx = db.load()
    target = {e["meta"]["target"] for e in idx.values()}.pop()
    crops = build_crops(idx, target)
    prompt = PROMPT.format(target)
    pos = sum(1 for _, l, _ in crops if l)
    majority = max(pos, len(crops) - pos) / len(crops)
    print(f"{len(crops)} distinct crops ({pos} yes / {len(crops) - pos} no), "
          f"target={target!r}")
    print(f"majority-class baseline {majority:.3f}\n")

    rows = {}
    for mp in BUDGETS:
        print(f"--- max_pixels={mp} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            v = make_verifier(max_pixels=mp)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            rows[str(mp)] = {"error": str(e)}
            continue

        try:
            for c, _, _ in crops[:WARMUP]:
                v(prompt, c)
            times, correct, degen = [], 0, 0
            for c, lbl, _ in crops:
                t0 = time.time()
                ans = v(prompt, c)
                times.append(time.time() - t0)
                if looks_degenerate(ans):
                    degen += 1
                    continue
                correct += (ans.strip().lower().startswith("y") == lbl)
            mean = sum(times) / len(times)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            acc = correct / len(crops)
            rows[str(mp)] = {"mean_s": round(mean, 3), "acc": round(acc, 3),
                             "degenerate": degen, "peak_mib": round(peak)}
            print(f"  {mean:.3f}s   acc {acc:.3f}   degenerate {degen}"
                  f"   peak VRAM {peak:.0f} MiB")
        except Exception as e:
            print(f"  FAILED during run: {e}")
            rows[str(mp)] = {"error": str(e)}
        finally:
            del v
            gc.collect()
            torch.cuda.empty_cache()
        json.dump(rows, open(OUT, "w"), indent=2)

    ok = {k: r for k, r in rows.items()
          if "error" not in r and r["degenerate"] == 0}
    print("\n" + "=" * 66)
    print("VISION BUDGET FRONTIER")
    print("=" * 66)
    print(f"{'max_pixels':>12}{'latency':>10}{'acc':>8}{'degen':>8}"
          f"{'peakVRAM':>11}")
    print("-" * 49)
    for k, r in rows.items():
        if "error" in r:
            print(f"{k:>12}   FAILED: {r['error'][:40]}")
        else:
            print(f"{k:>12}{r['mean_s']:>10.3f}{r['acc']:>8.3f}"
                  f"{r['degenerate']:>8}{r['peak_mib']:>11}")

    if not ok:
        print("\nNo budget ran cleanly. Keep 200704.")
        return

    cur = rows.get("200704")
    best = max(ok, key=lambda k: (ok[k]["acc"], -ok[k]["mean_s"]))
    print()
    if cur is None or "error" in cur:
        print(f"Control (200704) did not run; cannot compare. Keep current.")
        return

    gain = ok[best]["acc"] - cur["acc"]
    slow = ok[best]["mean_s"] - cur["mean_s"]
    if best == "200704" or gain <= 0.01:
        print(f"KEEP 200704. Best alternative ({best}) gains {gain:+.3f} "
              f"accuracy for {slow:+.3f}s.\nThat is inside the noise of "
              f"{len(crops)} crops, so the current budget sits at the knee --\n"
              f"which VALIDATES a setting that until now was only inherited "
              f"from a\nlatency emergency. No re-run needed.")
    else:
        print(f"ADOPT max_pixels={best}: {gain:+.3f} accuracy for "
              f"{slow:+.3f}s per call.\n"
              f"COST: every ablation number was measured at 200704, so "
              f"run_study.py,\ndecay_sweep.py and surround_sweep.py MUST be "
              f"re-run at the new value.\nDo not mix results from two "
              f"verifiers in one table.")
    degen_any = [k for k, r in rows.items()
                 if "error" not in r and r["degenerate"]]
    if degen_any:
        print(f"\n!! DEGENERATE OUTPUT at budgets {degen_any} -- the VRAM "
              f"ceiling is real\n   and nearer than assumed. Those budgets are "
              f"disqualified outright.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
