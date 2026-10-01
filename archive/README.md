# Archive: April to July 2026

Code from before the final system, kept so the project's history is visible. None of it is used by the final pipeline, and several paths and ports are specific to the Mac it ran on.

| File | Phase | What it was for |
|---|---|---|
| `early-code/mlx_test.py` | May | Running a small text model (Llama 3.2 1B, 4-bit) on the Mac's GPU through MLX to cut inference latency |
| `early-code/vision.py`, `early-code/vision_llava.py` | May | Sending a webcam frame to a local VLM through Ollama. LLaVA took about 68 s per call and failed the displacement tests, which is why the project moved to faster models |
| `early-code/vlm_benchmark.py`, `early-code/hardware_tests/moondream_spatial_probe.py` | June to July | Timing Moondream and asking it spatial questions. It produced no usable answers in 5 tries |
| `early-code/detection.py`, `early-code/spatial_test.py`, `early-code/hardware_tests/hsv_calibrate.py` | June | Red-cup tracking by HSV color threshold, before the switch to an open-vocabulary detector |
| `early-code/servo_arm.ino`, `early-code/hardware_tests/stage1_servo_test.ino`, `stage2_serial_test.*` | June | First servo firmware and the serial link from the laptop |
| `early-code/hardware_tests/stage3_camera_test.py`, `stage4_full_loop_test.py`, `early-code/test_cam-1.py` | June | Camera frame-rate check and the first camera-to-arm tracking loop |
| `mac-june-2026-code-snapshot.md` | June | Text snapshot of the modular June code (`~/avi` on the Mac): HSV tracking, serial control, per-servo calibration |

What each phase taught is written up in `docs/history/` and in the paper, Section 3.3.
