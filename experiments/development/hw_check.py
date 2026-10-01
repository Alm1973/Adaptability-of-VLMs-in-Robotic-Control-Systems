import sys
import time

import cv2

HOME = (90, 90)
NUDGE = 10
SETTLE_S = 1.2
DRIFT_TOLERANCE = 3


def ask(prompt, options):
    opts = "/".join(options)
    while True:
        try:
            got = input(f"  >>> {prompt} [{opts}]: ").strip().lower()
        except EOFError:
            raise SystemExit("\nABORT: no console to ask on. Run this "
                             "interactively -- it needs your eyes.")
        if got in options:
            return got
        if got == "abort":
            raise SystemExit("ABORT: requested by operator.")
        print(f"      answer one of {opts}, or 'abort'")


def check_camera():
    from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
    from camera import Camera
    print(f"\n[1/4] CAMERA  (config: index={CAMERA_INDEX} "
          f"{FRAME_WIDTH}x{FRAME_HEIGHT})")
    try:
        cam = Camera()
    except Exception as e:
        print(f"  FAIL: Camera() raised {type(e).__name__}: {e}")
        return None
    if not cam.warm:
        print("  FAIL: never produced a non-blank frame. Frames are black --")
        print("        do NOT run trials on them.")
        cam.release()
        return None
    cap = cam.cap

    ok, frame = cam.read_fresh()
    if not ok or frame is None:
        cam.release()
        print("  FAIL: warm, but read returned nothing.")
        return None

    h, w = frame.shape[:2]
    n, t0 = 30, time.time()
    fails, means = 0, []
    for _ in range(n):
        ok, f = cap.read()
        if not ok or f is None:
            fails += 1
        else:
            means.append(float(f.mean()))
    dt = time.time() - t0
    fps = n / dt if dt > 0 else 0
    mean = float(frame.mean())

    print(f"  resolution : {w}x{h}" +
          ("" if (w, h) == (FRAME_WIDTH, FRAME_HEIGHT)
           else f"   (CONFIG SAYS {FRAME_WIDTH}x{FRAME_HEIGHT})"))
    print(f"  capture    : {fps:.1f} fps over {n} frames, {fails} failed reads")
    print(f"  brightness : mean pixel {mean:.1f}"
          + (f"   (range {min(means):.0f}-{max(means):.0f} across the sweep)"
             if means else ""))
    if mean < 1.0:
        print("  FAIL: frame still black after warm-up. Stop and fix this.")
        cam.release()
        return None
    cv2.imwrite("hw_check_view.jpg", frame)
    print("  wrote hw_check_view.jpg -- open it and confirm the framing")

    seen = ask("Is that the controlled rig, framed the way you want?",
               ["y", "n"])
    cam.release()
    if seen == "n":
        print("  -> fix CAMERA_INDEX in config.py, or re-aim, then re-run.")
        return None
    return {"width": w, "height": h, "fps": round(fps, 1),
            "failed_reads": fails, "mean_brightness": round(mean, 1)}


def check_arm():
    print("\n[2/4] ARM  -- opening COM5 will RESET the Arduino and home the "
          "arm to 90/90.")
    go = ask("Arm clear of obstructions and you are watching?", ["y", "n"])
    if go == "n":
        raise SystemExit("ABORT: operator not ready.")

    from arm import Arm
    arm = Arm()
    time.sleep(SETTLE_S)
    print(f"  connected. tracked position: base={arm.base} tilt={arm.tilt}")

    result = {"base_axis": None, "tilt_axis": None, "channel1_joint": None,
              "returned_home": None}

    print(f"\n  Moving X by +{NUDGE} (base). Watch the arm.")
    input("  >>> press ENTER to move ")
    arm.update(NUDGE, 0)
    time.sleep(SETTLE_S)
    ans = ask("Which way did the camera swing?", ["left", "right", "none"])
    result["base_axis"] = ans
    arm.update(-NUDGE, 0)
    time.sleep(SETTLE_S)

    print(f"\n  Moving Y by +{NUDGE} (channel 1 -- the unknown one).")
    input("  >>> press ENTER to move ")
    arm.update(0, NUDGE)
    time.sleep(SETTLE_S)
    joint = ask("WHICH JOINT moved? (the one at the base of the arm = "
                "shoulder; the one holding the camera = wrist)",
                ["shoulder", "wrist", "none", "both"])
    direction = ask("Which way did the camera view move?",
                    ["up", "down", "none"])
    result["channel1_joint"] = joint
    result["tilt_axis"] = direction
    arm.update(0, -NUDGE)
    time.sleep(SETTLE_S)

    db, dt = arm.base - HOME[0], arm.tilt - HOME[1]
    if db or dt:
        arm.update(db, dt)
        time.sleep(SETTLE_S)
    drift = max(abs(arm.base - HOME[0]), abs(arm.tilt - HOME[1]))
    print(f"\n  tracked position after homing: base={arm.base} tilt={arm.tilt}"
          f"  (drift {drift})")
    if drift > DRIFT_TOLERANCE:
        raise SystemExit(f"ABORT: arm did not return home, drift={drift}")

    phys = ask("Is the arm physically back where it started?", ["y", "n"])
    result["returned_home"] = phys
    if phys == "n":
        print("  -> tracked position and physical position DISAGREE. Do not "
              "run trials until this is understood.")
    return result


def check_vlm():
    print("\n[3/4] VLM BACKEND  (llama-server, GPU)")
    import llm_backend
    t0 = time.time()
    up = llm_backend.ensure_running()
    print(f"  ensure_running() -> {up}  ({time.time()-t0:.1f}s)")
    if not up:
        print("  WARNING: falling back to ollama-CPU (~20-30x slower).")
        return {"backend": "ollama-cpu-fallback"}

    from camera import Camera
    cam = Camera()
    ok, frame = cam.read_fresh()
    cam.release()
    if not ok or frame is None:
        print("  could not grab a frame to test inference on.")
        return {"backend": "llama-server", "inference": None}

    t0 = time.time()
    out = llm_backend.vision_chat(
        "Describe what you see in one short sentence.", frame, max_tokens=20)
    dt = time.time() - t0
    degen = llm_backend._is_degenerate(out or "")
    print(f"  inference  : {dt:.2f}s")
    print(f"  output     : {str(out)[:70]!r}")
    print(f"  degenerate : {degen}  (guard is {'ACTIVE' if not degen else 'TRIPPED'})")
    return {"backend": "llama-server", "latency_s": round(dt, 2),
            "output": str(out)[:120], "degenerate": bool(degen)}


def check_detector():
    print("\n[4/4] DETECTOR")
    from config import USE_YOLO_TRACKER, YOLO_TARGET
    print(f"  USE_YOLO_TRACKER={USE_YOLO_TRACKER}   YOLO_TARGET={YOLO_TARGET!r}")
    if YOLO_TARGET.strip().lower() != "red cup":
        print("  ** YOLO_TARGET is not 'red cup'. The experiment targets the")
        print("     cup; this must be changed before any trial runs. **")
    return {"use_yolo": USE_YOLO_TRACKER, "target": YOLO_TARGET}


def main():
    print("=" * 72)
    print("hw_check.py -- pre-experiment verification.  THE ARM WILL MOVE.")
    print("=" * 72)
    report = {}
    report["camera"] = check_camera()
    if report["camera"] is None:
        raise SystemExit("\nSTOPPED: camera not usable.")
    report["detector"] = check_detector()
    report["arm"] = check_arm()
    report["vlm"] = check_vlm()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    import json
    print(json.dumps(report, indent=2))
    json.dump(report, open("hw_check.json", "w"), indent=2)
    print("\nwrote hw_check.json")

    ch1 = (report["arm"] or {}).get("channel1_joint")
    if ch1 in ("shoulder", "wrist"):
        print(f"\n>>> RESOLVED: PCA9685 channel 1 drives the {ch1.upper()}.")
        if ch1 == "shoulder":
            print("    SERVO_CALIBRATION.md is right about the joint, and the")
            print("    firmware is driving it with the WRIST's pulse range")
            print("    (140/380/580 vs the shoulder's 75/370/580). Conservative,")
            print("    not unsafe -- but the usable range is clipped at the low")
            print("    end and the constant names in the firmware are wrong.")
        else:
            print("    The firmware's pulse range is correct and")
            print("    SERVO_CALIBRATION.md's channel numbering is wrong:")
            print("    the wrist is on channel 1, not channel 3.")


if __name__ == "__main__":
    main()
