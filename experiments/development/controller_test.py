import random

from controller import Controller


class OldController:

    def __init__(self):
        self.kp = 0.015
        self.deadzone = 80
        self.max_step = 3

    def reset(self):
        pass

    def compute(self, ex, ey):
        out = []
        for e in (ex, ey):
            c = 0
            if abs(e) > self.deadzone:
                c = int(self.kp * e)
                c = max(-self.max_step, min(self.max_step, c))
            out.append(c)
        return out[0], out[1]


PX_PER_DEG = 42.0


def simulate(ctrl, err0, frames=80, noise=0.0, seed=0):
    rng = random.Random(seed)
    err = float(err0)
    cmds, errs = [], []
    for _ in range(frames):
        measured = err + rng.gauss(0, noise)
        cx, _ = ctrl.compute(int(measured), 0)
        err -= cx * PX_PER_DEG
        cmds.append(cx)
        errs.append(err)
    return cmds, errs


def metrics(ctrl, cmds, errs):
    jerk = (sum(abs(cmds[i] - cmds[i - 1]) for i in range(1, len(cmds)))
            / max(1, len(cmds) - 1))
    settle = None
    for i in range(len(errs)):
        if all(abs(e) <= ctrl.deadzone for e in errs[i:]):
            settle = i
            break
    return jerk, settle, errs[-1]


def run(label, err0, noise, seed=0):
    print(f"\n{label}  (start error {err0}px, sensor noise sigma={noise})")
    print(f"  {'controller':<12}{'jerk':>8}{'settle':>9}{'final err':>12}"
          f"{'moves':>8}")
    print("  " + "-" * 47)
    out = {}
    for name, ctrl in (("old", OldController()), ("new", Controller())):
        ctrl.reset()
        cmds, errs = simulate(ctrl, err0, noise=noise, seed=seed)
        j, s, f = metrics(ctrl, cmds, errs)
        moves = sum(1 for c in cmds if c != 0)
        out[name] = (j, s, f, moves)
        print(f"  {name:<12}{j:>8.3f}{str(s):>9}{f:>12.1f}{moves:>8}")
    return out


def main():
    print("=" * 60)
    print("CONTROLLER SMOOTHNESS -- old vs new, closed loop")
    print("=" * 60)

    a = run("1. LARGE STEP -- target appears far off centre", 500, 0.0)
    b = run("2. SMALL OFFSET -- just outside the deadzone", 95, 0.0)
    c = run("3. NOISY DETECTOR -- static target, jittery boxes", 0, 45.0)
    d = run("4. NOISY + OFFSET -- the realistic case", 260, 35.0)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    ok = True

    if c["new"][3] < c["old"][3]:
        print(f"  jitter rejection: new issues {c['new'][3]} moves vs "
              f"{c['old'][3]} on a STATIC target -- the limit cycle is gone")
    else:
        print(f"  !! jitter NOT improved ({c['new'][3]} vs {c['old'][3]} "
              f"moves on a static target)")
        ok = False

    if d["new"][0] < d["old"][0]:
        print(f"  smoothness: jerk {d['new'][0]:.3f} vs {d['old'][0]:.3f} "
              f"on the realistic trace")
    else:
        print(f"  !! jerk NOT improved ({d['new'][0]:.3f} vs "
              f"{d['old'][0]:.3f})")
        ok = False

    for label, r in (("large step", a), ("small offset", b)):
        ns, os_ = r["new"][1], r["old"][1]
        if ns is None:
            print(f"  !! {label}: new controller NEVER settles")
            ok = False
        elif os_ is not None and ns > os_ + 6:
            print(f"  !! {label}: settling regressed {os_} -> {ns} frames")
            ok = False
        else:
            print(f"  {label}: settles in {ns} frames "
                  f"(old {os_}), final error {r['new'][2]:.0f}px")

    print("\n" + ("PASS -- smoother without losing centring"
                  if ok else "FAIL -- do not put this on the robot"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
