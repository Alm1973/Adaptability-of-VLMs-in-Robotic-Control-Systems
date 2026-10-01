import sys
import time

import cv2

TARGET = "red cup"
WIN = "AVI live view  --  SPACE=set reference   r=clear   q=quit"


def main():
    target = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or TARGET
    from camera import Camera
    from yolo_tracker import YoloTracker

    print("opening camera...")
    cam = Camera()
    if not cam.warm:
        raise SystemExit("ABORT: camera never produced a non-blank frame.")
    print("loading detector...")
    tracker = YoloTracker(default_target=target)
    tracker.set_targets([target], allow_unreliable=True, quiet=True)
    print(f"\nlive. target={target!r}. press SPACE with the cup fully "
          f"visible to set the coverage reference, q to quit.\n")

    ref_area = None
    seen = miss = 0
    t0 = time.time()
    fps = 0.0
    n = 0

    while True:
        ok, frame = cam.read_fresh()
        if not ok or frame is None:
            continue
        n += 1
        if n % 10 == 0:
            fps = 10.0 / max(time.time() - t0, 1e-6)
            t0 = time.time()

        det = tracker.find(frame, target)
        h, w = frame.shape[:2]
        view = frame.copy()

        if det:
            seen += 1
            x, y, bw, bh = det["box"]
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 230, 0), 3)
            cv2.circle(view, det["center"], 5, (0, 230, 0), -1)
        else:
            miss += 1

        bar = 128
        cv2.rectangle(view, (0, 0), (w, bar), (0, 0, 0), -1)
        if det:
            cv2.putText(view, "CUP DETECTED", (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 3)
            cv2.putText(view, f"confidence {det.get('conf', '?')}"
                              f"    area {det['area']}", (14, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (200, 255, 200), 2)
        else:
            cv2.putText(view, "NOT DETECTED", (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(view, "the pipeline would call this a loss", (14, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (180, 180, 255), 2)

        if ref_area:
            if det:
                cov = max(0.0, 1.0 - det["area"] / ref_area)
                if cov < 0.70:
                    txt, col = f"coverage ~{cov*100:.0f}%  PARTIAL", (0, 220, 220)
                elif cov < 0.85:
                    txt, col = (f"coverage ~{cov*100:.0f}%  HEAVY, still seen",
                                (0, 165, 255))
                else:
                    txt, col = (f"coverage ~{cov*100:.0f}%  NEAR THE LIMIT",
                                (0, 100, 255))
            else:
                txt, col = "FULLY COVERED (detector lost it)", (0, 0, 255)
            cv2.putText(view, txt, (14, 112), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, col, 2)
        else:
            cv2.putText(view, "press SPACE with the cup clear to enable "
                              "coverage", (14, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 2)

        tot = seen + miss
        cv2.putText(view, f"{fps:4.1f} fps   detected {100*seen/max(tot,1):.0f}%"
                          f" of {tot}   mean {frame.mean():.0f}",
                    (w - 470, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (220, 220, 220), 2)

        cv2.imshow(WIN, view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' '):
            if det:
                ref_area = det["area"]
                print(f"[ref] reference area set to {ref_area} "
                      f"(cup treated as 0% covered)")
            else:
                print("[ref] no detection -- cannot set a reference on a "
                      "frame where the cup is not found")
        if k == ord('r'):
            ref_area = None
            print("[ref] cleared")

    cam.release()
    cv2.destroyAllWindows()
    tot = seen + miss
    print(f"\n{tot} frames, cup detected in {seen} ({100*seen/max(tot,1):.0f}%)")


if __name__ == "__main__":
    main()
