import csv
import json
import os
import shutil
import sys
import tempfile

import final_experiment as fx

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}")
        print(f"        want {want!r}")
        FAILED.append(name)


def frame(phase, present, believes, edge=False, status="CONFIRMED"):
    return {"phase": phase, "present": present, "believes": believes,
            "edge": edge, "status": status, "t": 0.0, "vlm_calls": 0,
            "raw": "r.jpg", "annotated": "a.jpg"}


print("\n[1] counterbalancing")
order = fx.build_order(["a", "b", "c"], reps=4, seed=1)
check("total trials = scenarios x reps", len(order), 12)
counts = {s: sum(1 for n, _, _ in order if n == s) for s in ("a", "b", "c")}
check("each scenario appears equally", counts, {"a": 4, "b": 4, "c": 4})

blocks_ok = all(
    sorted(n for n, r, _ in order if r == rep) == ["a", "b", "c"]
    for rep in (1, 2, 3, 4))
check("every rep is a full block of all scenarios", blocks_ok, True)
check("order is not simply grouped by scenario",
      [n for n, _, _ in order] != ["a"] * 4 + ["b"] * 4 + ["c"] * 4, True)

cl = fx.build_order(["a", "b", "c"], reps=4, seed=1, clutters=("none", "high"))
check("clutter doubles the trial count", len(cl), 24)
check("each clutter level appears equally",
      {c: sum(1 for _, _, x in cl if x == c) for c in ("none", "high")},
      {"none": 12, "high": 12})
check("every scenario x clutter cell is filled",
      sorted({(n, c) for n, _, c in cl}),
      sorted([(n, c) for n in "abc" for c in ("none", "high")]))
n_blocks = 4 * 2
switches = sum(1 for i in range(1, len(cl)) if cl[i][2] != cl[i - 1][2])
check("restages never exceed the block count", (switches + 1) <= n_blocks,
      True)
check("restages far fewer than one per trial", (switches + 1) < len(cl), True)
check("clutter does actually alternate", switches >= 1, True)
check("clutter constant within a block",
      all(len({c for _, _, c in cl[i:i + 3]}) == 1
          for i in range(0, len(cl), 3)), True)
first_of_each_rep = [next(c for n, r, c in cl if r == rep)
                     for rep in (1, 2, 3, 4)]
check("clutter level order varies across reps",
      len(set(first_of_each_rep)) > 1, True)
check("same seed reproduces the order",
      fx.build_order(["a", "b", "c"], 4, 1), order)
check("different seed gives a different order",
      fx.build_order(["a", "b", "c"], 4, 2) != order, True)

print("\n[2] pre-registered success criteria")
check("high accuracy, no false belief -> PASS", fx.classify(0.90, 0), "PASS")
check("exactly at the PASS threshold -> PASS",
      fx.classify(fx.PASS_ACC, 0), "PASS")
check("mid accuracy -> PARTIAL", fx.classify(0.55, 0), "PARTIAL")
check("exactly at the PARTIAL threshold -> PARTIAL",
      fx.classify(fx.PARTIAL_ACC, 0), "PARTIAL")
check("low accuracy -> FAIL", fx.classify(0.10, 0), "FAIL")
check("perfect accuracy but ONE false belief -> FAIL",
      fx.classify(1.00, 1), "FAIL")
check("PARTIAL accuracy with false belief -> FAIL",
      fx.classify(0.55, 3), "FAIL")
check("no scoreable frames -> FAIL", fx.classify(None, 0), "FAIL")

print("\n[3] belief scoring against capture-time ground truth")

perfect = ([frame("baseline", True, True)] * 4 +
           [frame("disrupt", True, True)] * 6 +
           [frame("recover", True, True)] * 4)
s = fx.score_trial(perfect)
check("perfect trial: accuracy 1.0", s["during_disruption_acc"], 1.0)
check("perfect trial: no false belief", s["false_belief_frames"], 0)
check("perfect trial: nothing lost", s["lost_while_present"], 0)
check("perfect trial verdict", fx.classify(s["during_disruption_acc"],
                                           s["false_belief_frames"]), "PASS")

ghost = ([frame("baseline", True, True)] * 4 +
         [frame("disrupt", True, True)] * 6 +
         [frame("recover", False, True)] * 6)
s = fx.score_trial(ghost)
check("ghost trial: false belief counted", s["false_belief_frames"], 6)
check("ghost trial is a FAIL despite decent accuracy",
      fx.classify(s["during_disruption_acc"], s["false_belief_frames"]),
      "FAIL")

dropped = ([frame("baseline", True, True)] * 4 +
           [frame("disrupt", True, False)] * 6 +
           [frame("recover", True, True)] * 4)
s = fx.score_trial(dropped)
check("dropped trial: lost_while_present counted",
      s["lost_while_present"], 6)
check("dropped trial: no false belief", s["false_belief_frames"], 0)
check("accuracy uses the DISRUPT window only",
      s["during_disruption_acc"], 0.0)

lenient = ([frame("baseline", True, True)] * 4 +
           [frame("disrupt", True, False)] * 6 +
           [frame("recover", True, True)] * 30)
check("a long clean recover cannot inflate accuracy",
      fx.score_trial(lenient)["during_disruption_acc"], 0.0)

edged = ([frame("baseline", True, True)] * 2 +
         [frame("disrupt", True, False, edge=True)] * 5 +
         [frame("disrupt", True, True)] * 5)
s = fx.score_trial(edged)
check("edge frames excluded from grading", s["during_disruption_acc"], 1.0)
check("edge frames counted separately", s["edge_frames"], 5)
check("edge frames still counted in the total", s["total_frames"], 12)

spur = ([frame("baseline", True, True)] * 3 +
        [frame("recover", False, True, status="CONFIRMED")] * 3)
check("spurious reacquisition detected",
      fx.score_trial(spur)["spurious_reacquisition"], True)
notspur = ([frame("baseline", True, True)] * 3 +
           [frame("recover", False, False, status="MISSING")] * 3)
check("correct MISSING is not spurious",
      fx.score_trial(notspur)["spurious_reacquisition"], False)

print("\n[3b] edge window scales with phase length")
for sec in (1.0, 1.5, 2.0, 6.0, 12.0):
    e = fx.edge_window(sec)
    check(f"phase {sec}s leaves a scoreable middle", (sec - 2 * e) > 0, True)
check("real setting keeps the full 1.0s margin", fx.edge_window(6.0), 1.0)
check("smoke setting shrinks the margin", fx.edge_window(1.5), 0.375)

print("\n[4] output files")
tmp = tempfile.mkdtemp(prefix="fxtest_")
try:
    rows = [{"trial_no": 1, "scenario": "occlude_hand", "rep": 1,
             "clutter": "none", "verdict": "PASS", "wall_clock_s": 21.0,
             "failure_mode": None, "vlm_calls": 2, "trial_dir": "trials/t1",
             "frames": perfect, **fx.score_trial(perfect)},
            {"trial_no": 2, "scenario": "removal", "rep": 1,
             "clutter": "none", "verdict": "FAIL", "wall_clock_s": 22.5,
             "failure_mode": None, "vlm_calls": 3, "trial_dir": "trials/t2",
             "frames": ghost, **fx.score_trial(ghost)}]
    fx.write_results(tmp, rows)

    with open(os.path.join(tmp, "results.csv"), encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    check("csv has one row per trial", len(got), 2)
    check("csv carries the primary metric",
          got[1]["false_belief_frames"], "6")
    check("csv links row -> first image", got[0]["first_frame"], "r.jpg")
    check("csv has no nested frame blob",
          "frames" in got[0], False)

    loaded = json.load(open(os.path.join(tmp, "results.json")))
    check("json keeps the full per-frame trajectory",
          len(loaded[0]["frames"]), 14)

    cfg = {"created": "test", "hypothesis_primary": "H",
           "independent_live": ["disruption"],
           "independent_offline": ["ablation"],
           "dependent_primary": "false_belief_frames",
           "dependent_secondary": ["acc"], "held_constant": ["lighting"],
           "success_criteria": {"PASS": "p", "PARTIAL": "q", "FAIL": "r"},
           "scenarios": ["occlude_hand"], "reps": 1, "clutter": "none",
           "seed": 1, "edge_seconds": 1.0}
    fx.write_readme(tmp, cfg, rows)
    readme = open(os.path.join(tmp, "README.md"), encoding="utf-8").read()
    check("readme states the hypothesis", "H" in readme, True)
    check("readme states the criteria", "PASS" in readme, True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[5] scenario definitions")
for name, phases in fx.SCENARIOS.items():
    check(f"{name}: three phases", len(phases), 3)
    check(f"{name}: instructions are non-empty",
          all(len(i) > 20 for _, _, i in phases), True)
for name in ("removal", "substitution", "identity_swap"):
    check(f"{name}: ends with the target ABSENT",
          fx.SCENARIOS[name][-1][1], False)
for name in ("control", "occlude_hand", "occlude_object"):
    check(f"{name}: ends with the target PRESENT",
          fx.SCENARIOS[name][-1][1], True)
check("control never disrupts anything",
      all(p for _, p, _ in fx.SCENARIOS["control"]), True)

print("\n[6] props and clutter definitions")
for sc, prop in fx.REQUIRES_PROP.items():
    check(f"{sc}: prop named", len(prop) > 5, True)
    check(f"{sc}: prop appears in its instructions",
          any(prop.split()[0].lower() in i.lower()
              or prop.split()[-1].lower().strip("()") in i.lower()
              for _, _, i in fx.SCENARIOS[sc]), True)
for lvl, desc in fx.CLUTTER_LEVELS.items():
    check(f"clutter '{lvl}': countable definition", len(desc) > 30, True)
check("clutter levels include a bare control", "none" in fx.CLUTTER_LEVELS,
      True)

print("\n" + "=" * 64)
if FAILED:
    print(f"{len(FAILED)} CHECK(S) FAILED:")
    for n in FAILED:
        print(f"  - {n}")
    sys.exit(1)
print("ALL CHECKS PASSED -- the harness logic is sound.")
print("Remaining risk is hardware and staging, which only a real run finds.")
