# Adaptability of VLMs in Robotic Control Systems

## Project Title
Robotic System for Object Detection

## Project Description
For the current project, we have designed a vision based robot control system which involves the use of a camera, VLM, and a robotic arm controlled by the microcontroller. The system is supposed to identify objects within a workspace, get their location from the processed images, and actuate physical movements through the use of an Arduino actuation system and guide the robotic arm there.

## Research Question
Can a locally-run vision-language model, operating within a hybrid perception-reasoning pipeline where OpenCV handles spatial localization, reliably recover from task-disrupting situations — occlusion, environmental change, and abnormal/unrecognized targets — during an object verification task, by adapting its recovery strategy and repositioning the camera autonomously? (See [`docs/research-question.md`](docs/research-question.md) for the full revision history and hypothesis.)

## Current Hardware Setup
- Arduino Uno 
- USB camera / webcam mounted above workspace
- Robotic arm (servo-based, 4 DOF)
- Servo motors for movement
- USB connection between computer and Arduino
- Laptop running Python (OpenCV + Moondream)
- PCA9685 16-Channel PWM Servo Driver
- UBEC 

See [`docs/hardware.md`](docs/hardware.md) for hardware photos.

## System Architecture

The system uses a **hybrid OpenCV + VLM pipeline**: OpenCV handles deterministic perception (HSV color masking → centroid tracking) and drives the arm directly while tracking is healthy; a locally-run VLM (Qwen2.5-VL:3B via Ollama) is only invoked for higher-level recovery reasoning when tracking is lost. See [`docs/project-database.md`](docs/project-database.md) for the full architecture decision record, servo calibration data, benchmarking results, and design rationale.

```
OpenCV (perception)
        │
        ▼
   Tracking OK? ──Yes──▶ PID correction ──┐
        │                                  │
        No                                 ▼
        │                          State Machine
        ▼                                  │
 Qwen2.5-VL (recovery reasoning)  ─────────┘
                                            ▼
                                    Arduino → Servos
```

## How to Run the Code

1. Flash `avi_control.ino` to the Arduino (PCA9685-based servo driver).
2. Activate your Python environment: `source ~/robotenv/bin/activate`
3. Install dependencies (once): `pip install opencv-python numpy pyserial ollama`
4. Start Ollama and pull the model: `ollama serve` then `ollama pull qwen2.5vl:3b`
5. Run the system: `python main.py`

## Documentation

- [`docs/project-database.md`](docs/project-database.md) — architecture decision record, servo calibration, benchmarks
- [`docs/hardware.md`](docs/hardware.md) — hardware photos
- [`docs/system-schematic.md`](docs/system-schematic.md) — full wiring + data flow diagram
- [`docs/research-question.md`](docs/research-question.md) — revised research question, hypothesis, title options
- [`docs/vlm-spatial-reasoning-review.md`](docs/vlm-spatial-reasoning-review.md) — literature review + OpenCV/VLM division of labor
- [`docs/grid-testing-protocol.md`](docs/grid-testing-protocol.md) — full-grid detection accuracy test protocol
- [`code/hardware_tests/`](code/hardware_tests/) — staged, isolated tests for closing the hardware loop
- [`lab-notebook.md`](lab-notebook.md) — dated experiment log
