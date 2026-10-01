import json
import random
import sys

import cv2

EXPLORE_EPSILON = 0.4
STOCHASTIC = ("random", "vlm_eps")
TRIALS_PER_START = 5
MAX_STEPS = 14
WIN_W, WIN_H = 560, 340
STEP = 140
OUT = "search_study.json"

MAX_PER_DIRECTION = 5
START_OFFSET_X = WIN_W // 2 + 130
START_OFFSET_Y = WIN_H // 2 + 90

DELTA = {"left": (-STEP, 0), "right": (STEP, 0),
         "up": (0, -STEP), "down": (0, STEP)}


def view(world, cx, cy):
    H, W = world.shape[:2]
    x = max(0, min(W - WIN_W, cx - WIN_W // 2))
    y = max(0, min(H - WIN_H, cy - WIN_H // 2))
    return world[y:y + WIN_H, x:x + WIN_W], x, y


def target_visible(box, x, y, frac=0.6):
    x1, y1, x2, y2 = box
    ix1, iy1 = max(x1, x), max(y1, y)
    ix2, iy2 = min(x2, x + WIN_W), min(y2, y + WIN_H)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    area = (x2 - x1) * (y2 - y1)
    return area > 0 and (iw * ih) / area >= frac


def legal_dirs(world, cx, cy, tried):
    H, W = world.shape[:2]
    out = []
    for d, (dx, dy) in DELTA.items():
        if tried[d] >= MAX_PER_DIRECTION:
            continue
        nx, ny = cx + dx, cy + dy
        _, ox, oy = view(world, cx, cy)
        _, nox, noy = view(world, nx, ny)
        if (nox, noy) == (ox, oy):
            continue
        out.append(d)
    return out


def run_episode(strategy, world, box, start, pipe=None, rng=None):
    cx, cy = start
    tried = {d: 0 for d in DELTA}
    chosen = []
    for step in range(MAX_STEPS):
        frame, x, y = view(world, cx, cy)
        if target_visible(box, x, y):
            return True, step, chosen
        legal = legal_dirs(world, cx, cy, tried)
        if not legal:
            return False, step, chosen

        if strategy == "vlm":
            want = pipe.search_direction(frame, tried)
            d = want if want in legal else min(legal, key=lambda k: tried[k])
        elif strategy == "vlm_eps":
            if rng.random() < EXPLORE_EPSILON:
                d = rng.choice(legal)
            else:
                want = pipe.search_direction(frame, tried)
                d = want if want in legal else min(legal,
                                                   key=lambda k: tried[k])
        elif strategy == "sweep":
            order = ["left", "right", "up", "down"]
            d = next((k for k in order if k in legal), legal[0])
        else:
            d = rng.choice(legal)

        chosen.append(d)
        tried[d] += 1
        dx, dy = DELTA[d]
        cx, cy = cx + dx, cy + dy

    frame, x, y = view(world, cx, cy)
    return target_visible(box, x, y), MAX_STEPS, chosen


def main():
    world_path = sys.argv[1] if len(sys.argv) > 1 else "_live_now.jpg"
    target = " ".join(sys.argv[2:]).strip() or "red cup"

    world = cv2.imread(world_path)
    if world is None:
        raise SystemExit(f"ABORT: cannot read {world_path}")
    H, W = world.shape[:2]
    print(f"world {W}x{H}, window {WIN_W}x{WIN_H}, target {target!r}")

    import disruption_bench as db
    found = db._target_box(world, target)
    if not found:
        raise SystemExit(f"ABORT: {target!r} not in {world_path}")
    conf, box = found
    tx, ty = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    print(f"target at conf {conf:.2f}, centre ({tx},{ty})")

    starts = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        sx = min(W - 1, max(0, tx + dx * START_OFFSET_X))
        sy = min(H - 1, max(0, ty + dy * START_OFFSET_Y))
        _, vx, vy = view(world, sx, sy)
        need_x = abs(tx - (vx + WIN_W // 2)) / STEP
        need_y = abs(ty - (vy + WIN_H // 2)) / STEP
        if (not target_visible(box, vx, vy)
                and need_x <= MAX_PER_DIRECTION
                and need_y <= MAX_PER_DIRECTION):
            starts.append((sx, sy))
    if not starts:
        raise SystemExit("ABORT: no start is both out of view and reachable -- "
                         "adjust START_OFFSET / STEP / MAX_PER_DIRECTION")
    print(f"{len(starts)} valid start positions "
          f"(out of view AND reachable in <={MAX_PER_DIRECTION} steps)\n")

    from recovery_pipeline import RecoveryPipeline
    from run_study import make_verifier
    print("loading verifier...")
    pipe = RecoveryPipeline(target, tracker=None, verifier=make_verifier())

    results = {}
    for strat in ("vlm", "vlm_eps", "sweep", "random"):
        rng = random.Random(0)
        before = pipe.vlm_calls
        wins, steps_when_found, all_steps = 0, [], []
        trials = TRIALS_PER_START if strat in STOCHASTIC else 1
        for s in starts:
            for _ in range(trials):
                ok, steps, chosen = run_episode(strat, world, box, s,
                                                pipe=pipe, rng=rng)
                all_steps.append(steps)
                if ok:
                    wins += 1
                    steps_when_found.append(steps)
        n = len(all_steps)
        results[strat] = {
            "found": wins, "of": n,
            "success_rate": round(wins / n, 3) if n else None,
            "mean_steps_when_found": (round(sum(steps_when_found) /
                                            len(steps_when_found), 2)
                                      if steps_when_found else None),
            "vlm_calls": pipe.vlm_calls - before,
        }
        r = results[strat]
        print(f"  {strat:<8} found {r['found']}/{r['of']}  "
              f"({r['success_rate']})  mean steps "
              f"{r['mean_steps_when_found']}  vlm_calls={r['vlm_calls']}")

    json.dump(results, open(OUT, "w"), indent=2)
    print("\n" + "=" * 68)
    print("DOES VLM GUIDANCE BEAT AN UNINFORMED SEARCH?")
    print("=" * 68)
    print(f"{'strategy':<10}{'found':>10}{'rate':>9}{'mean steps':>13}"
          f"{'vlm calls':>12}")
    print("-" * 54)
    for k in ("vlm", "vlm_eps", "sweep", "random"):
        d = results[k]
        print(f"{k:<10}{d['found']}/{d['of']:<7}{str(d['success_rate']):>9}"
              f"{str(d['mean_steps_when_found']):>13}{d['vlm_calls']:>12}")

    v, ve, rd = results["vlm"], results["vlm_eps"], results["random"]
    print("\n" + "-" * 68)
    print("READING THIS TABLE")
    if ve["success_rate"] >= rd["success_rate"] and \
            ve["success_rate"] > v["success_rate"]:
        print("  vlm_eps beat plain vlm AND matched or beat random: the "
              "epsilon-greedy\n  fix worked, and VLM guidance is worth keeping "
              "as a prior.")
    elif ve["success_rate"] > v["success_rate"]:
        print("  vlm_eps beat plain vlm but not random: exploration fixed the "
              "COMMITMENT\n  failure, yet the VLM's suggestions still add "
              "nothing over chance. The\n  honest conclusion is that the gain "
              "came from the randomness, not the model.")
    else:
        print("  vlm_eps did NOT beat plain vlm. The epsilon-greedy change is "
              "not\n  justified by this measurement and should not be "
              "reported as a fix.")
    print("\n  Note the vlm calls column: exploration steps skip the model, so "
          "vlm_eps\n  is also the cheaper policy. If accuracy ties, cost "
          "breaks the tie.")
    print("\nIf vlm does NOT beat sweep and random, the VLM's scene "
          "understanding\nis not contributing to recovery -- and "
          "'VLM-guided recovery' is a claim\nthe data does not support. "
          "That is a real result, not a failure.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
