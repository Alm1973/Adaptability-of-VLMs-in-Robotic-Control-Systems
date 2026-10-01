import sys

import cv2

import llm_backend
from camera import Camera

MAX_TOKENS = 300
MAX_HISTORY = 4
FRAME_DUMP = "_ask_frame.jpg"


def grab(cam):
    ok, frame = cam.read_fresh()
    if not ok or frame is None:
        raise SystemExit("ABORT: camera read failed")
    if frame.mean() < 1.0:
        raise SystemExit("ABORT: camera returned a blank frame")
    return frame


def answer(frame, question, history):
    return llm_backend.vision_chat(
        question, frame, max_tokens=MAX_TOKENS, history=history[-MAX_HISTORY:])


def main():
    one_shot = " ".join(sys.argv[1:]).strip()

    cam = Camera()
    if not cam.warm:
        cam.release()
        raise SystemExit("ABORT: camera never produced a non-blank frame.")

    if not llm_backend.ensure_running():
        cam.release()
        raise SystemExit("ABORT: llama-server not healthy")

    frame = grab(cam)
    history = []

    if one_shot:
        out = answer(frame, one_shot, history)
        print(f"\n{out if out else '(no answer -- backend failed)'}")
        cam.release()
        return

    print("=" * 68)
    print("Ask the robot about what it sees.  look / save / reset / quit")
    print("=" * 68)
    try:
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                continue
            low = q.lower()

            if low in ("quit", "exit"):
                break
            if low == "look":
                frame = grab(cam)
                history = []
                print("  [fresh frame, conversation reset]")
                continue
            if low == "save":
                cv2.imwrite(FRAME_DUMP, frame)
                print(f"  [wrote {FRAME_DUMP}]")
                continue
            if low == "reset":
                history = []
                print("  [conversation cleared, same frame]")
                continue

            out = answer(frame, q, history)
            if out is None:
                print("  (backend failed -- no answer)")
                continue
            print(f"\n{out}")
            history.append((q, out))
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        print("\nbye")


if __name__ == "__main__":
    main()
