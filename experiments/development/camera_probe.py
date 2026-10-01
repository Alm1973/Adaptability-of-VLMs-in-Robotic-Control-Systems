import cv2

from config import CAMERA_INDEX

BACKENDS = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]

print(f"configured CAMERA_INDEX = {CAMERA_INDEX}\n")
working = []
for i in range(4):
    for backend, name in BACKENDS:
        cap = None
        try:
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"  index {i} {name:6s} -> FRAME {frame.shape}")
                working.append((i, name))
            else:
                print(f"  index {i} {name:6s} -> opened but no frame")
        except Exception as e:
            print(f"  index {i} {name:6s} -> error {e}")
        finally:
            if cap is not None:
                cap.release()

print()
if not working:
    print("NO WORKING CAMERA. Either the device is unplugged, or another "
          "process holds it (teleop.py / a browser tab streaming /video / "
          "Windows Camera app). Nothing to fix in code until it returns.")
else:
    print(f"working: {working}")
    idxs = sorted({i for i, _ in working})
    if CAMERA_INDEX not in idxs:
        print(f"CONFIGURED INDEX {CAMERA_INDEX} IS STALE -> use {idxs[0]} "
              f"(set CAMERA_INDEX in config.py)")
    else:
        print(f"configured index {CAMERA_INDEX} is fine; the earlier failure "
              f"was likely a transient hold by another process")
