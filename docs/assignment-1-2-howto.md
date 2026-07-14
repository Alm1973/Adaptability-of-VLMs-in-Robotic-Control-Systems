# How to Run Assignments 1 & 2

These two assignments need your hands on the actual hardware — here's exactly what to do, in order.

## Assignment 1: Close the Hardware Loop

### Step 1 — Order parts (do this today, before anything else)

- **PCA9685 servo driver** — confirm the exact model you had (mentor's email says "PCA6-something," likely a typo for PCA9685). Search Amazon for "PCA9685 16-Channel PWM Servo Driver," ~$6–10.
- **Logitech C270 webcam** — ~$20, also on Amazon.

Order both now — everything else in this assignment can happen while you wait for them to arrive, except the steps that need the actual hardware.

### Step 2 — Stage 1: servo driver test (needs: PCA9685, Arduino, 1 servo)

1. Wire the PCA9685 to the Arduino: SDA→SDA, SCL→SCL, GND→GND (shared ground is critical), V+ to your UBEC servo power rail (not the Arduino's 5V pin).
2. Connect one servo to PCA9685 channel 0.
3. Open `code/hardware_tests/stage1_servo_test.ino` in the Arduino IDE, select your board/port, and upload it.
4. Open the Serial Monitor (Tools → Serial Monitor, 115200 baud). You should see the servo sweep smoothly from 0° to 180° and back, with angle values printing.
5. **Pass condition:** smooth motion, no jitter or stalling at the edges. If it jitters near an edge, narrow `SERVO_MIN`/`SERVO_MAX` in the sketch and re-upload.
6. Log the result in `lab-notebook.md` before moving on.

### Step 3 — Stage 2: serial communication test (needs: Stage 1 working)

1. Flash `code/hardware_tests/stage2_serial_test.ino` to the Arduino (replaces Stage 1's sketch).
2. On your laptop, run `ls /dev/cu.*` (Mac) to find your Arduino's port name.
3. Open `code/hardware_tests/stage2_serial_test.py` and update the `PORT` variable at the top to match.
4. Run `python code/hardware_tests/stage2_serial_test.py`.
5. At the `>` prompt, type `PING` — you should see `PONG` printed back within about a second. Then try `ROTATE_BASE_LEFT_15` and `ROTATE_BASE_RIGHT_15` — the servo should visibly move each time.
6. Log the result.

### Step 4 — Stage 3: camera test (needs: Logitech C270 delivered)

1. Plug in the C270.
2. Run `python code/hardware_tests/stage3_camera_test.py` from inside `code/` (it imports `detection.py`, so run it from that folder or add `code/` to your path).
3. A window should open showing the live feed with an FPS counter and detection status overlay. Point a red/yellow/blue/green object at the camera and confirm it gets boxed and labeled with a grid position.
4. Note the FPS shown, and write down anything odd (focus issues, poor detection in your room's lighting, etc.) in the lab notebook — the mentor specifically asked for this.

### Step 5 — Stage 4: full loop test (needs: Stages 1–3 all passing)

1. Flash your full `avi_control.ino` (or reuse `stage2_serial_test.ino` if it's sufficient) to the Arduino.
2. Make sure Ollama is running (`ollama serve`) and you have a model pulled (`ollama pull moondream` or `qwen2.5vl:3b`).
3. In `code/hardware_tests/stage4_full_loop_test.py`, update `PORT` to match your Arduino, and set `HARDCODE_ACTION = "ROTATE_BASE_LEFT_15"` for your very first run — this proves the mechanical loop works without waiting on the VLM.
4. Run it. You should see detection, a hardcoded action being sent, and the servo moving, all printed to the terminal each cycle.
5. Once that works, set `HARDCODE_ACTION = None` to let the VLM actually decide the action, and re-run.
6. **This is the Assignment 1 milestone** — message your mentor once you see one full cycle complete, even if the VLM's chosen action isn't smart yet.

## Assignment 2: Grid Testing

This only needs your laptop + webcam — you can do it in parallel with waiting on Assignment 1's parts, or right after Stage 3 passes.

1. Open `docs/grid-testing-protocol.md` — it has the full table template.
2. Set up your workspace so you can consistently place an object at 5 grid positions: center, top-left, top-right, bottom-left, bottom-right.
3. For each position, run 3 rounds (centered, shifted-left, shifted-right within that square):
   - Place the object.
   - Run `code/detection.py` directly (`python code/detection.py`) — it opens a live window and prints/overlays the detected grid label.
   - Compare the printed label to where you actually placed the object.
   - Mark correct/incorrect in the table in `docs/grid-testing-protocol.md`.
4. If a trial is wrong, immediately write down what happened in the failure log at the bottom of that same doc — don't wait, per the mentor's note about the earlier untracked failure.
5. Once all 15+ trials are done, fill in the accuracy percentages and answer the three pattern-analysis questions at the bottom of the doc (edges vs. center, corner asymmetry, boundary effects).

That table and analysis is real experimental data for your paper — I can't generate it for you, but the protocol and scoring is already set up so it's just placing objects and recording what the script says.
