# Adaptability of VLMs in Robotic Control Systems

## Project Title
Robotic System for Object Detection

## Project Description
In this project, the vision based robot control system has been designed where the camera, VLM, and robotic arm with the microcontroller have been used. The robot control system will be responsible for object detection in the workspace, localization of the object by usingg images processed and actuating of the physical movement by the help of Arduino actuation system.

## Research Question
Is it possible that a locally hosted vision language model, embedded in a perception-reasoning pipeline that uses OpenCV to do spatial localization, be able to recover from disruptions in the task of object verification due to occlusion, changes in the environment, and unexpected objects?

## Current Hardware Setup
- Arduino Uno 
- USB camera / webcam mounted above workspace
- Robotic arm (servo-based, 4 DOF)
- Servo motors for movement
- USB connection between computer and Arduino
- Laptop running Python (OpenCV + Moondream)
- PCA9685 16-Channel PWM Servo Driver
- UBEC 

See [`docs/Hardware.md`](docs/Hardware.md) for hardware photos.

## System Architecture

It is implemented using a **hybrid OpenCV + VLM workflow**: OpenCV is responsible for deterministic perception (HSV masking + centroid tracking) as well as directly controlling the arm in case of healthy tracking, while a locally executed VLM (Qwen2.5-VL:3B from Ollama) is called upon for recovery reasoning if tracking fails

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

1. Flash `avi_control.ino` to the Arduino.
2. Activate your Python environment: `source ~/robotenv/bin/activate`
3. Install dependencies (once): `pip install opencv-python numpy pyserial ollama`
4. Start Ollama and pull the model: `ollama serve` then `ollama pull qwen2.5vl:3b`
5. Run the system: `python main.py`

## Documentation

- [`docs/project-database.md`](docs/project-database.md) 
- [`docs/Hardware.md`](docs/Hardware.md) 
- [`docs/system-schematic.md`](docs/system-schematic.md) 
- [`docs/research-question.md`](docs/research-question.md)
- [`docs/vlm-spatial-reasoning-review.md`](docs/vlm-spatial-reasoning-review.md) 
- [`code/hardware_tests/`](code/hardware_tests/) 
- [`lab-notebook.md`](lab-notebook.md) 
