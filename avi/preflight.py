import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

FAIL, WARN = [], []


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")
    FAIL.append(msg)


def warn(msg):
    print(f"  [warn] {msg}")
    WARN.append(msg)


print("=" * 72)
print("PREFLIGHT")
print("=" * 72)

print("\n[1] repo state")
try:
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        warn(f"uncommitted changes:\n{dirty}")
    else:
        ok("working tree clean")
    head = subprocess.run(["git", "log", "-1", "--oneline"],
                          capture_output=True, text=True).stdout.strip()
    ok(f"HEAD {head}")
except Exception as e:
    warn(f"git unavailable: {e}")

print("\n[2] scripts parse and import")
import ast
for f in ("final_experiment.py", "score_run.py", "live_view.py",
          "recovery_pipeline.py", "run_study.py", "yolo_tracker.py",
          "camera.py", "arm.py"):
    try:
        ast.parse(open(f, encoding="utf-8").read())
        ok(f"{f} parses")
    except Exception as e:
        bad(f"{f}: {e}")

try:
    import final_experiment as fx
    ok(f"final_experiment imports ({len(fx.SCENARIOS)} scenarios)")
except Exception as e:
    bad(f"final_experiment import: {e}")
    print("\nCANNOT CONTINUE")
    sys.exit(1)

print("\n[3] experiment design")
order = fx.build_order(fx.DEFAULT_SCENARIOS, 3, 20260901, ("none", "high"))
ok(f"{len(order)} trials, {len(fx.DEFAULT_SCENARIOS)} scenarios x 3 reps x 2 "
   f"clutter levels")
per_cell = {}
for n, r, c in order:
    per_cell[(n, c)] = per_cell.get((n, c), 0) + 1
if len(set(per_cell.values())) == 1:
    ok(f"every scenario x clutter cell has n={next(iter(per_cell.values()))}")
else:
    bad(f"unbalanced cells: {per_cell}")
switches = sum(1 for i in range(1, len(order))
               if order[i][2] != order[i - 1][2]) + 1
ok(f"{switches} clutter restages (not {len(order)})")
e = fx.edge_window(fx.PHASE_SECONDS)
ok(f"phase {fx.PHASE_SECONDS}s, edge {e}s/end -> "
   f"{fx.PHASE_SECONDS - 2*e}s scoreable per phase")
if fx.PHASE_SECONDS - 2 * e <= 0:
    bad("no scoreable window per phase")

print("\n[4] resources")
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    free = int(out.split()[0])
    if free < 4000:
        bad(f"only {free} MiB VRAM free -- another VLM is probably loaded. "
            f"The run needs ~3.8 GB.")
    else:
        ok(f"{free} MiB VRAM free")
except Exception as ex:
    warn(f"could not read GPU: {ex}")

try:
    free_gb = shutil.disk_usage(".").free / 1e9
    need = len(order) * 3 * fx.PHASE_SECONDS * 10 * 2 * 0.1 / 1000
    if free_gb < need * 2:
        warn(f"{free_gb:.1f} GB free, run needs ~{need:.1f} GB")
    else:
        ok(f"{free_gb:.0f} GB free (run needs ~{need:.1f} GB)")
except Exception as ex:
    warn(f"disk check: {ex}")

print("\n[5] arm port (listed, NOT opened)")
try:
    import serial.tools.list_ports as lp
    ports = [p.device for p in lp.comports()]
    if "COM5" in ports:
        ok(f"COM5 present ({ports})")
    else:
        bad(f"COM5 missing; saw {ports}")
except Exception as ex:
    bad(f"serial: {ex}")

print("\n[6] camera and detector")
cam = None
try:
    from camera import Camera
    from yolo_tracker import YoloTracker
    cam = Camera()
    if not cam.warm:
        bad("camera never produced a non-blank frame")
    else:
        ok("camera warm")
    t0, n, means = time.time(), 20, []
    for _ in range(n):
        okr, f = cam.read_fresh()
        if okr and f is not None:
            means.append(float(f.mean()))
    fps = n / (time.time() - t0)
    ok(f"{fps:.1f} fps, brightness {min(means):.0f}-{max(means):.0f}")
    if min(means) < 1.0:
        bad("black frames present")

    tr = YoloTracker(default_target=fx.TARGET)
    tr.set_targets([fx.TARGET], allow_unreliable=True, quiet=True)
    hits, confs = 0, []
    for _ in range(10):
        okr, f = cam.read_fresh()
        if not okr or f is None:
            continue
        d = tr.find(f, fx.TARGET)
        if d:
            hits += 1
            confs.append(d.get("conf", 0))
    if hits >= 8:
        ok(f"target {fx.TARGET!r} detected {hits}/10, conf "
           f"{min(confs):.2f}-{max(confs):.2f}")
    elif hits > 0:
        warn(f"target detected only {hits}/10 -- restage before running")
    else:
        bad(f"target {fx.TARGET!r} NOT detected. Baseline would be invalid.")
except Exception as ex:
    bad(f"camera/detector: {type(ex).__name__}: {ex}")

print("\n[7] pipeline constructs with the exact args the run passes")
verifier = None
try:
    from recovery_pipeline import RecoveryPipeline
    from run_study import CONDITIONS, make_fast_verifier, score_episode
    for c in ("D_full", "E_no_vlm"):
        if c in CONDITIONS:
            ok(f"condition {c} defined")
        else:
            bad(f"condition {c} MISSING -- score_run needs it")
    print("       loading verifier (~25s)...")
    verifier = make_fast_verifier()
    ok("verifier loaded")
    p = RecoveryPipeline(fx.TARGET, tracker=tr, verifier=verifier,
                         use_detector=True, use_opencv=True, use_vlm=True,
                         use_state=True)
    ok(f"pipeline built, status={p.status}")
    okr, f = cam.read_fresh()
    t0 = time.time()
    p.step(f)
    ok(f"step() ran in {(time.time()-t0)*1000:.0f} ms -> {p.status}, "
       f"believes={p.believes_present()}, vlm_calls={p.vlm_calls}")
    p.notify_self_motion()
    ok("notify_self_motion() available (camera_pose needs it)")
except Exception as ex:
    bad(f"pipeline: {type(ex).__name__}: {ex}")

print("\n[8] score_run end-to-end on a fabricated run (the untested path)")
try:
    import cv2
    import score_run
    tmp = tempfile.mkdtemp(prefix="preflight_run_")
    tdir = os.path.join(tmp, "trials", "trial_001_removal")
    os.makedirs(tdir)
    frames, gt = [], []
    for i in range(12):
        okr, f = cam.read_fresh()
        if not okr or f is None:
            continue
        pth = os.path.join(tdir, f"f{i:03d}_raw.jpg")
        cv2.imwrite(pth, f)
        phase = "baseline" if i < 4 else ("disrupt" if i < 8 else "recover")
        frames.append({"phase": phase, "present": phase != "recover",
                       "edge": i in (0, 11), "raw": os.path.relpath(pth, tmp),
                       "annotated": "", "t": i * 0.1, "status": "CONFIRMED",
                       "believes": True, "vlm_calls": 0})
    json.dump([{"trial_no": 1, "scenario": "removal", "rep": 1,
                "clutter": "none", "verdict": "PASS", "frames": frames}],
              open(os.path.join(tmp, "results.json"), "w"))

    eps, skipped = score_run.build_episodes(tmp)
    if len(eps) == 1:
        ok(f"build_episodes rebuilt {len(eps)} episode from results.json")
    else:
        bad(f"build_episodes returned {len(eps)} episodes, expected 1")
    ep = next(iter(eps.values()))
    nones = sum(1 for g in ep["gt"] if g["present"] is None)
    if nones == 2:
        ok(f"edge frames handed over as GT None ({nones})")
    else:
        bad(f"edge handling wrong: {nones} None, expected 2")
    ok(f"meta disrupt_start={ep['meta']['disrupt_start']} "
       f"recover_start={ep['meta']['recover_start']}")

    p2 = RecoveryPipeline(fx.TARGET, tracker=tr, verifier=verifier,
                          use_detector=True, use_opencv=True, use_vlm=True,
                          use_state=True)
    r = score_episode(p2, ep, ep["frames"])
    ok(f"score_episode ran: durAcc={r['during_disruption_acc']} "
       f"false={r['false_belief_frames']} lost={r['lost_while_present']}")
    shutil.rmtree(tmp, ignore_errors=True)
except Exception as ex:
    bad(f"score_run path: {type(ex).__name__}: {ex}")

try:
    if cam:
        cam.release()
except Exception:
    pass

print("\n" + "=" * 72)
if FAIL:
    print(f"{len(FAIL)} BLOCKER(S) -- do not start the run:")
    for m in FAIL:
        print(f"  - {m}")
    sys.exit(1)
if WARN:
    print(f"GO, with {len(WARN)} warning(s):")
    for m in WARN:
        print(f"  - {m}")
else:
    print("ALL CHECKS PASSED -- clear to run.")
sys.exit(0)
