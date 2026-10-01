import cv2

from controller import Controller
from heldout_gt import HELDOUT_GT
from yolo_tracker import YoloTracker, UNRELIABLE_CLASSES, MIN_CONSECUTIVE_HITS

TARGET = "computer mouse"
FRAME_W, FRAME_H = 1280, 720


def check_unreliable_guard():
    print("=" * 70)
    print("1. UNRELIABLE_CLASSES guard")
    print("=" * 70)
    t = YoloTracker()
    for bad in ("water bottle", "bottle"):
        try:
            t.set_targets([bad])
            print(f"  FAIL: {bad!r} was accepted -- guard did not fire")
        except ValueError as e:
            print(f"  ok: {bad!r} refused -> {str(e)[:88]}...")
    return t


def check_tracking(t):
    print("\n" + "=" * 70)
    print(f"2. Detection + servo commands (target={TARGET!r})")
    print("=" * 70)
    t.set_targets([TARGET])
    ctrl = Controller()
    cx0, cy0 = FRAME_W // 2, FRAME_H // 2

    print(f"  frame centre ({cx0},{cy0}); deadzone={ctrl.deadzone} "
          f"kp={ctrl.kp} max_step={ctrl.max_step}")
    print(f"  {'frame':<26}{'gt':>3}{'det':>6}{'conf':>7}"
          f"{'errX':>7}{'errY':>7}{'moveX':>7}{'moveY':>7}")
    print("  " + "-" * 68)

    for frame_name in sorted(HELDOUT_GT):
        img = cv2.imread(frame_name)
        if img is None:
            print(f"  {frame_name}: unreadable")
            continue
        gt = HELDOUT_GT[frame_name].get(TARGET, "?")

        r = None
        for i in range(MIN_CONSECUTIVE_HITS):
            r = t.find(img, target=TARGET)
            if i == 0 and r is not None:
                print(f"  WARNING: streak gate did not suppress first hit")
        t.reset_history()
        for _ in range(MIN_CONSECUTIVE_HITS):
            r = t.find(img, target=TARGET)

        if r is None:
            print(f"  {frame_name:<26}{gt:>3}{'--':>6}{'--':>7}"
                  f"{'--':>7}{'--':>7}{'0':>7}{'0':>7}   NO MOVE")
            continue

        cx, cy = r["center"]
        errX, errY = cx - cx0, cy - cy0
        moveX, moveY = ctrl.compute(errX, errY)
        print(f"  {frame_name:<26}{gt:>3}{'yes':>6}{r['conf']:>7}"
              f"{errX:>7}{errY:>7}{moveX:>7}{moveY:>7}")
        t.reset_history()


def check_suppression(t):
    print("\n" + "=" * 70)
    print("3. Suppression zone blocks a detection")
    print("=" * 70)
    frame_name = "c2_10_home.jpg"
    img = cv2.imread(frame_name)
    if img is None:
        print("  frame unreadable, skipping")
        return
    t.reset_history()
    t.suppression_zones = []
    r = None
    for _ in range(MIN_CONSECUTIVE_HITS):
        r = t.find(img, target=TARGET)
    if r is None:
        print("  no baseline detection to suppress, skipping")
        return
    print(f"  baseline: {r['label']} at {r['center']} conf={r['conf']}")
    t.add_suppression_zone(r["center"])
    t.reset_history()
    r2 = None
    for _ in range(MIN_CONSECUTIVE_HITS):
        r2 = t.find(img, target=TARGET)
    if r2 is None or r2["center"] != r["center"]:
        print(f"  ok: suppressed -> {r2['center'] if r2 else None}")
    else:
        print("  FAIL: same centre returned despite suppression zone")


if __name__ == "__main__":
    tr = check_unreliable_guard()
    check_tracking(tr)
    check_suppression(tr)
    print("\nNo serial port opened. No arm command sent. arm.py never imported.")
