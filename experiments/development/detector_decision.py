import json
import sys

import cv2

OUT = "detector_decision.json"
SHEET = "_detector_decision.jpg"

COCO80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}

PROBES = [
    ("keyboard", "coco"),
    ("cup", "coco"),
    ("mouse", "coco"),
    ("bottle", "coco"),
    ("red cup", "attribute"),
    ("blue cup", "attribute-absent"),
    ("green backlit keyboard", "attribute"),
    ("black computer mouse", "attribute"),
    ("coconut water carton", "open"),
    ("screwdriver", "open"),
    ("usb hub", "open"),
    ("computer monitor", "open"),
    ("elephant", "absent"),
    ("bicycle", "absent"),
]

CONF_FLOOR = 0.10


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def probe(frame, name):
    from yolo_tracker import YoloTracker
    t = YoloTracker(default_target=name)
    t.set_targets([name], allow_unreliable=True, quiet=True)
    res = t.model.predict(frame, conf=CONF_FLOOR, verbose=False)[0]
    best = None
    for b in res.boxes:
        if t.model.names[int(b.cls.item())] != name:
            continue
        c = float(b.conf.item())
        if best is None or c > best[0]:
            best = (c, [int(v) for v in b.xyxy[0].tolist()])
    return best


def main():
    frame_path = sys.argv[1] if len(sys.argv) > 1 else "_live_now.jpg"
    frame = cv2.imread(frame_path)
    if frame is None:
        raise SystemExit(f"ABORT: cannot read {frame_path}")
    print(f"frame: {frame_path} ({frame.shape[1]}x{frame.shape[0]})\n")

    rows = {}
    for name, category in PROBES:
        got = probe(frame, name)
        rows[name] = {
            "category": category,
            "in_coco": name in COCO80,
            "fired": got is not None,
            "conf": round(got[0], 3) if got else 0.0,
            "box": got[1] if got else None,
        }
        r = rows[name]
        print(f"  {name:<24} [{category:<17}] coco={str(r['in_coco']):<5} "
              f"fired={str(r['fired']):<5} conf={r['conf']}")

    open_names = [n for n, c in PROBES if c in ("open", "attribute")]
    open_fired = [n for n in open_names if rows[n]["fired"]]
    coco_expressible = [n for n in open_names if n in COCO80]

    EXCLUSIVE = [("red cup", "blue cup")]
    fired = {n: rows[n]["box"] for n, _ in PROBES
             if rows[n]["fired"] and rows[n]["box"]}
    pairs = []
    for a, b in EXCLUSIVE:
        if a in fired and b in fired:
            pairs.append((a, b, _iou(fired[a], fired[b])))
    identical = [(a, b, v) for a, b, v in pairs if v >= 0.9]

    red, blue = rows.get("red cup", {}), rows.get("blue cup", {})
    grounded = red.get("fired") and not blue.get("fired")
    partial = (red.get("fired") and blue.get("fired")
               and red.get("conf", 0) > blue.get("conf", 0) * 1.5)

    sheet = frame.copy()
    for name, r in rows.items():
        if r["box"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(sheet, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(sheet, f"{name} {r['conf']:.2f}", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    cv2.imwrite(SHEET, sheet)

    print("\n" + "=" * 72)
    print("DECISION: FIXED-CLASS vs OPEN-VOCABULARY")
    print("=" * 72)
    print(f"\n1. COVERAGE of arbitrary names")
    print(f"   open-vocab fired on {len(open_fired)}/{len(open_names)} "
          f"non-COCO / attribute-qualified names")
    print(f"   a fixed COCO head could express {len(coco_expressible)}"
          f"/{len(open_names)} of them AT ALL")
    print(f"   -> {', '.join(open_fired) if open_fired else '(none)'}")

    print(f"\n2. PROMPT SENSITIVITY  (is it reading the name, or just "
          f"returning what is salient?)")
    print(f"   {len(pairs)} MUTUALLY EXCLUSIVE name pairs both fired; "
          f"{len(identical)} returned the SAME box (IoU>=0.9)")
    for a, b, v in pairs:
        mark = "!!" if v >= 0.9 else "  "
        print(f"     {mark} {a!r} vs {b!r} -> IoU {v:.2f}")
    if identical:
        print("   -> names that CANNOT describe the same object return the "
              "same box.\n      The detector is matching the noun and "
              "ignoring the qualifier, so it\n      cannot by itself serve a "
              "target whose name carries an attribute.")
    else:
        print("   -> mutually exclusive names map to different boxes: the "
              "prompt is\n      genuinely read, qualifiers included.")

    print(f"\n3. ATTRIBUTE GROUNDING  ('red cup' present, 'blue cup' absent)")
    print(f"   red cup  fired={red.get('fired')} conf={red.get('conf')}")
    print(f"   blue cup fired={blue.get('fired')} conf={blue.get('conf')}")
    if grounded:
        print("   -> colour attributes are grounded: the absent colour does "
              "not fire.")
    elif partial:
        print("   -> partially grounded: both fire, but the correct colour "
              "is clearly\n      stronger. Usable with a confidence margin, "
              "not as a hard gate.")
    else:
        print("   -> NOT grounded: the detector cannot tell the colours "
              "apart. Colour-named\n      targets must be verified by "
              "something else -- which is precisely the\n      job the "
              "impostor episode assigns to the VLM.")

    fp = [n for n, c in PROBES if c == "absent" and rows[n]["fired"]]
    print(f"\n4. FALSE POSITIVES on absent objects: "
          f"{fp if fp else 'none'}")

    print("\n" + "-" * 72)
    print("CONCLUSION")
    print("  A fixed COCO head cannot express most targets a user would type, "
          "so it\n  cannot satisfy decision 2 at any accuracy. Open-vocab is "
          "the only option\n  that meets the requirement; the measurements "
          "above record HOW WELL it\n  meets it, and where the pipeline must "
          "compensate.")
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT} and {SHEET}")


if __name__ == "__main__":
    main()
