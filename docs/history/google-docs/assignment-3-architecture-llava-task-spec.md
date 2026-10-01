# Assignment 3 — Architecture A vs B, LLaVA latency benchmark, task & perturbation spec (May 2026)

> Source: Google Doc "Assigment3" · (private Google Drive link removed)
> Created 2026-05-19 · Last modified 2026-05-20 · Imported 2026-09-28 (verbatim)
> The Google Doc "meeting" (private Google Drive link removed) 2026-05-20) is an AI-generated condensed summary of the "Task Specification" tab below — not imported separately.
> Repo link in doc: github.com/Alm1973/robot-vision-project — this is the old name of Adaptability-of-VLMs-in-Robotic-Control-Systems (same commit history).

# Tab 1
https://github.com/Alm1973/robot-vision-project/tree/main/code

**Assignment 1**
Occlusion:
A: Pixy2 camera would struggle with a complex environment since it is trained on specific color, signatures, and shapes. If during the experiment the camera is blurred the pixy2 camera will lose functionality entirely and misread position and size of it. In combination with this the LLM would struggle to reason due to the hardware constraints and couldn't infer as well.
B: Occlusion is handled better in the system due to the VLM receiving full camera images and the ability to reason without any camera constraints. This would allow us to focus on the inferability of the Ai and not make the robot independent.

Environmental change:
A: lighting shifts for example can distort color the Pixy2 camera receives and the the camera itself, relying on color signatures of object, would make the process come to a halt entirely due to lack of perception. This perception cant be solved for since using a more advanced camera would work against the hardware constraints.
B: The VLM being able to analyze full visual data and also being able to handle variation in certain variables like lighting, object placement, and color will make it more understandable. This would ultimately make it more dynamic and be more reliable in dynamic conditions.

**Assignment 2**
(b) Architecture Decision and Justification
I am choosing Architecture B, because it allows the model to be capable of analyzing images taken by a camera and making inferences about the work environment. Architecture B suits my experiment since it allows the use of VLM which is more suitable for identifying unexpected situations and adapting to them. As opposed to the Pixy2 sensor, the Visual Language Model recognizes unusual objects and is able to interpret more complicated situations and interpret them based on other variables. Architecture makes the experiment more complex and results in increased latency since the computing process will not take place on a local computer but rather in a cloud API or on a laptop but it is better because the LLM isn't constrained on a local system like how it is in architecture A. Additionally, in Architecture B, it may be difficult to find out where the problem arises since there is a chance of perception, communication, or reasoning errors.. The necessary components are Arduino, servos, a USB webcam, and a laptop with VLM installed.

(c) Compute Plan for Architecture B
The hardware components include a USB camera which is attached to the computer where the Python code is run. The frames would be taken from the camera using OpenCV and sent to a VLM, running in Ollama.
The analysis done by the VLM will make the VLM choose from a set of pre-defined actions like ROTATE_LEFT_15 or TILT_UP_10. This action command will be sent through serial communication via USB from the Python code to the Arduino microcontroller. The servos will be controlled by the Arduino according to the received command.

Model Selection and Latency Benchmark:
(a) -I chose LLaVA

Trial 1 — Latency: 62.502 seconds
Output: In the image provided, there is a person who appears to be a young man with short hair, looking directly at the camera. He has on a black top and is against a plain background that looks like a wall. The photo seems to have a vintage or retro filter applied to it, giving it a slightly faded or old appearance.
Given the context you provided, I would choose CONFIRM_TARGET, as there do not appear to be any other objects in the image that need manipulation or rotation.

Trial 2 — Latency: 64.294 seconds
Output: The image is too blurry to provide a detailed description of objects within it. However, if I were controlling a robotic arm and the objective was to perform an action on the visible portion of the image, here's what I would do:
ROTATE_LEFT_15
Please note that this is a hypothetical action based on the assumption that there is something tangible in the image to interact with. Since the image is very blurry and lacks clear details, the actual actions required may vary greatly depending on what specific object or target needs to be manipulated.

Trial 3 — Latency: 74.511 seconds
Output: The image provided appears to be a photo of a blank screen or monitor displaying a black background with a white rectangle in the center. There are no objects visible in this image to describe. However, if we were to assume that this is a representation of a target for a robotic arm operation, I could choose one of the actions based on what I would do next:
Given the context, if this blank screen represents the target, I would opt for "CONFIRM_TARGET" as it aligns with the typical use of such a setup in robotic applications where confirming the target is an important step before proceeding with the operation.

Trial 4 — Latency: 58.659 seconds
Output: Based on the image you've provided, which shows a person's lower body and feet with no clear interaction with objects or any actions being performed with a robotic arm, I would select "CONFIRM_TARGET." This action could be used if the target is to confirm the current position or alignment of the robotic arm.

Trial 5 — Latency: 78.792 seconds
Output: There are no visible objects in the image that you have provided. The image is too blurry and indistinct to discern any specific items or details. Based on what I can see, it appears to be a silhouette of a person standing against a dark background, but there is not enough detail to determine if this is an actual scene or some form of digital manipulation.
As for the requested action, since no specific objects are present in the image, none of the provided actions (ROTATE_LEFT_15, ROTATE_RIGHT_15, TILT_UP_10, TILT_DOWN_10, CONFIRM_TARGET, REPORT_FAILURE) can be applied. If you have a clearer image or more context, I could assist you further.

FINAL RESULTS
Average latency: 67.751 seconds
Min: 58.659
Max: 78.792

(b) Model Behavior
| Scenario | Expected Behavior | Actual Output | Correct? |
| :- | :- | :- | :- |
| Centered | STOP | STOP | Yes |
| Top-left | MOVE_UP/LEFT | STOP | No |
| Bottom-right | MOVE_DOWN/RIGHT | STOP | No |

The experiment with the model LLaVa was done against a plain white background using a centering task with a calculator positioned at different locations of the camera frame. the model was able to recognize the case of centering and understand the prompt given and issued the correct STOP command when the calculator was centered. But for the other two scenarios the model failed to evaluate the correct output to achieve centering by issuing a STOP command when movement was needed. The model is capable of recognizing the presence of an object but it fails to understand displacement in its environment.

(c) Report
Architecture B with LLaVA has a good ability to reason from the whole image allowing the system to deal with issues such as occlusion and variations in lighting that would be a problem for a Pixy2 camera in Architecture A. Although the latency is really high at an average of around 67.8 seconds, the model does partially succeed in reasoning about higher level understanding as seen from the outputs. It successfully recognizes centered targets and gives proper commands in some cases. The issue is that the model fails at correctly moving in specific directions which shows that the model cannot reason with displacements in its environment. As for feasibility, the speed is currently too slow for real time control systems.

**-Fixes?**
- Switch to a smaller VLM
- Reduce model prompt length
- Disable unnecessary
- Try Phi-3 Vision, Gemma Vision
- Run inference on a GPU instead of CPU
- Increase RAM availability
- Reduce frame rate
- Keep model loaded dont reload again
- Trade image resolution vs speed

# Task Specification
### (a) System Task Definition
The test takes place on a flat matte-white workspace measuring 150 cm × 150 cm in order to reduce any reflective distractions. A fixed 3 × 3 grid sectioned off nine areas (front, middle, back × left, center, right) is used for object positioning. The robotic arm is installed at the rear-center of the workspace facing forwards towards the grid's center point.

It has three degrees of movement operated by servos: base rotation (left/right), shoulder tilt (up/down), and wrist/camera tilt. The USB webcam (720p or 1080p) is attached to the end effector positioned about 25-35 cm above the workspace offering a 60-70° field of view. In the resting position, the camera points at the middle grid cell.

Four predetermined objects are used in both architectures: red star block, yellow star block (both about 4 cm with identical shapes but different colors), blue cube (about 3 cm), and green cylinder (about 4 cm). In Architecture [B], the VLM will be able to recognize all objects.

One of the following commands is generated by the program: rotate/tilt command, CONFIRM_TARGET, or REPORT_FAILURE. All these commands are performed by servos under control of an Arduino board. The loop goes on until some stopping condition is met.

Only in cases when the CONFIRM_TARGET command is sent to the system without any human involvement can we consider this trial as successful. In all other cases, where the REPORT_FAILURE command was sent or the system exceeded the time limit of 60 seconds we consider it a failure.

**(b) Perturbation Definitions**
The output from the system is either direction correction (rotate/tilt), CONFIRM_TARGET, or REPORT_FAILURE. This is done through Arduino control of servos and the loop repeats itself.

The occlusion test will be performed in three ways. First, occlusion will be created by covering 25% of the object using a card positioned on one of its corners. Second, occlusion will be created by covering 50% of the object using a horizontal card, thus showing only the lower part of the object. Third, the occlusion will be dynamic, with a hand passing in front of the object every five seconds.

The environmental variability will be considered based on two different aspects. First, lighting will be varied into three conditions: regular overhead lighting, darkened environment, and glare lighting due to reflection of light from a lamp. Second displacement of objects will occur after the initial process starts either by moving the object 2-3 cm inside the same grid cell or moving the object to another grid cell for example from center to middle right.
