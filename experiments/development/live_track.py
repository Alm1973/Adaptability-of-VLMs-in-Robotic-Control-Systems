import sys
import time

import cv2

from adaptive_detector import (Suppression, configure_validated,
                               detect_candidates, verify)
from camera import Camera

SETTLE_READS = 18
VERIFY_COOLDOWN_S = 3.0
WINDOW = "live track"


def settle(cam, n=SETTLE_READS):
    f = None
    for _ in range(n):
        ok, cur = cam.read()
        if ok and cur is not None:
            f = cur
        time.sleep(0.04)
    return f


def main():
    args = sys.argv[1:]
    max_frames = 0
    if "--frames" in args:
        i = args.index("--frames")
        max_frames = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    headless = "--no-window" in args
    args = [a for a in args if a != "--no-window"]
    query = " ".join(args).strip()
    if not query:
        try:
            query = input("what should I track? > ").strip()
        except EOFError:
            return
    if not query:
        return

    print(f"tracking: {query!r}")
    cam = Camera()
    frame = settle(cam)
    if frame is None:
        print("camera gave no frame")
        cam.release()
        return

    print("configuring detector (VLM)...")
    cfg, cfg_time = configure_validated(frame, query)
    if cfg is None:
        print("could not configure -- the VLM could not find that object "
              "in the current view. Point the camera at it and retry.")
        cam.release()
        return
    print(f"  cues={cfg.cues}  colour={cfg.color_name!r}  "
          f"config={cfg_time:.1f}s")
    print("tracking... (q or ESC to quit)\n")

    supp = Suppression()
    last_verify = 0.0
    verified_box = None
    verify_calls = 0
    frames = 0
    cv_total = 0.0
    t_start = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            frames += 1

            t0 = time.time()
            cands = detect_candidates(frame, cfg, suppression=supp)
            cv_dt = time.time() - t0
            cv_total += cv_dt

            top = cands[0] if cands else None
            now = time.time()
            if top and now - last_verify > VERIFY_COOLDOWN_S:
                ok_v, _ = verify(frame, top["box"], query)
                verify_calls += 1
                last_verify = now
                if ok_v:
                    verified_box = top["box"]
                else:
                    supp.add(top["center"])
                    verified_box = None

            disp = frame.copy()
            if top:
                x, y, w, h = top["box"]
                col = (0, 255, 0) if verified_box else (0, 200, 255)
                cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                label = f"{query}" if verified_box else "candidate"
                cv2.putText(disp, label, (x, max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            fps = frames / max(1e-6, time.time() - t_start)
            hud = [f"{query}  cues={','.join(cfg.cues)}",
                   f"opencv {cv_dt*1000:5.1f}ms   {fps:4.1f} fps   "
                   f"VLM calls: {verify_calls}"]
            if not headless:
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 62), (0, 0, 0), -1)
                for i, line in enumerate(hud):
                    cv2.putText(disp, line, (10, 24 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WINDOW, disp)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
            if max_frames and frames >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        elapsed = time.time() - t_start
        print(f"\nframes            : {frames}")
        print(f"elapsed           : {elapsed:.1f}s  ({frames/max(1e-6,elapsed):.1f} fps)")
        if frames:
            print(f"opencv per frame  : {cv_total/frames*1000:.1f} ms")
        print(f"VLM config calls  : 3 (cues + locate + colour)")
        print(f"VLM verify calls  : {verify_calls}")
        print(f"VLM calls / frame : "
              f"{(verify_calls)/max(1,frames):.3f}  "
              f"(baseline would be 3.0)")


if __name__ == "__main__":
    main()
