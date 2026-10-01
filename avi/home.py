import sys
import time

import serial

PORT = "COM5"
BAUD = 115200
RESET_WAIT = 2.5


def main():
    print(f"[HOME] opening {PORT} (this resets the Uno and re-centres)...")
    try:
        s = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"[HOME] FAILED to open {PORT}: {e}")
        print("       If this says 'device not functioning', unplug the "
              "Arduino, wait 5s, replug. If that fails, Device Manager -> "
              "Ports -> Arduino Uno -> Disable then Enable.")
        return 1

    time.sleep(RESET_WAIT)
    try:
        s.write(b"X:0 Y:0\n")
        s.flush()
    except Exception as e:
        print(f"[HOME] port opened but write failed: {e}")
        s.close()
        return 1

    s.close()
    print("[HOME] done -- arm centred at (90, 90), port released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
