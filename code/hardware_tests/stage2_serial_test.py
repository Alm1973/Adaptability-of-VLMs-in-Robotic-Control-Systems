
import sys
import time

import serial

PORT = "/dev/cu.usbmodem1101"
BAUD = 115200


def main():
    print(f"Connecting to {PORT} @ {BAUD}...")
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # let the Arduino reset after the serial connection opens

    # Drain the READY banner
    while arduino.in_waiting:
        print("Arduino:", arduino.readline().decode(errors="ignore").strip())

    print("Connected. Type a command (PING / ROTATE_BASE_LEFT_15 / "
          "ROTATE_BASE_RIGHT_15), or 'quit' to exit.")

    while True:
        cmd = input("> ").strip()
        if cmd.lower() == "quit":
            break
        if not cmd:
            continue

        arduino.write((cmd + "\n").encode())
        time.sleep(0.2)

        while arduino.in_waiting:
            print("Arduino:", arduino.readline().decode(errors="ignore").strip())

    arduino.close()


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        print("Run `ls /dev/cu.*` and update PORT at the top of this file.")
        sys.exit(1)

# PASS CRITERIA (log in lab-notebook.md):
#   - PING returns PONG within ~1 second.
#   - ROTATE_BASE_LEFT_15 / RIGHT_15 visibly move the servo and print
#     "OK angle=<value>" back.
