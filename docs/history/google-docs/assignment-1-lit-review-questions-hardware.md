# Assignment 1 — Literature review, candidate questions, hardware inventory, roadmap (April 2026)

> Source: Google Doc "Assigment1" · (private Google Drive link removed)
> Created 2026-04-05 · Last modified 2026-04-14 · Imported 2026-09-28 (verbatim text; two embedded images not imported)
> Superset of the three "Assigment1*.pdf" files already in this project. Early phase (Raspberry Pi / TinyML framing) — superseded by later docs.

# Literature Rev.
1. Reddi, Vijay Janapa, et al. *TinyML Course — Lecture Series*. Harvard University / TinyMLx, YouTube, uploaded by TinyMLx. www.youtube.com/watch?v=AWxJY_hS_Ac&list=PLJ-4lGVfhv34qW0r4dt_qXyY3-HezeGUo&index=1.
   This lecture, with Dr.Evgani Gousev, talked about how machine learning interacts with sensors and how we connect the physical world to the digital one. Tiny machine learning's advantages are that it is energy efficient(10-100 mAh), less processing power is needed(10,100 mHz), and less memory (less than 1MB). Towards the end it described a few examples such as tiny ML on voice and MEM sensors.
2. Reddi, Vijay Janapa, et al. "*[Video Title]*." *TinyML Course Lecture Series*, Harvard University / TinyMLx, *[Upload Date]*, www.youtube.com/watch?v=enUuGgPX_HY.
   This lecture was led by Daniel Pau. Artificial intelligence is just an approximation. Hardware needs capabilities to initiate multiple neural networks. [image]
3. Edje Electronics. "How to Run YOLO Object Detection Models on the Raspberry Pi." *YouTube*, 1 year ago, www.youtube.com/watch?v=z70ZrSZNi-8.
   The first step was to set up raspberry pi connected to the internet. Then set up ultralytics(YOLO). After that you have to download the necessary python libraries and either custom train the AI or use an already compiled data set from online.
4. Brenpoly. "I Made a Real BMO Local AI Agent with a Raspberry Pi and Ollama." *YouTube*, 2 months ago, www.youtube.com/watch?v=l5ggH-YhuAw.
   This video explains how he made a locally run ai from a raspberry pi using Olama which allowed him to run a LLM locally with minimal fees attached to it. He also put a conversational ai onto the raspberry pi without needing the cloud.
5. Yoshida, et al. "From Text to Motion: Grounding GPT-4 in a Humanoid Robot 'Alter3.'" *Frontiers in Robotics and AI*, vol. 12, 2025, doi.org/10.3389/frobt.2025.1581110.
   This paper talks about using a LLM in a two stage process where the first stage is accepting user input into the LLm and then converting it into a prompt. The second step receives the prompt from the first step and converts the existing prompt into a segment of python code which gets sent to the alter 3 robot discussed in the paper. To be able to write this code the second instance of the use of GPT-4 has to have a detailed description of what the bot is what what it can be able to integrate its code into. For example the connection and what available sensors it has the positions of the components and the measurements and dimensions of the bot itself.
6. *Ollama: Run AI Models Locally for Secure & Cost-Effective Embedded Development.* Design News, 2 June 2025, www.designnews.com/artificial-intelligence/ollama-run-ai-models-locally-for-secure-cost-effective-embedded-development.
   This article talks about Ollama which is a platform which allows to run certain models of ai locally. Ollama can run on your computer but it is not specified wether it can run on microcontrollers in the article.
   Benefits:
   - Security: Your code and project data never leave your machine.
   - Cost efficiency: There's no usage-based pricing—it's free to use.
   - Speed: Responses are faster with no API call overhead or internet dependency. (This does depend on your local machine's GPU, though.)
   - Control: You decide which models to run, when to update them, and how they're integrated into your workflow.

Summary questions:
- What can TinyML do well on a microcontroller? What are the limits of its capabilities?
  -TinyML is very efficient on a microcontroller specifically and has less requirements to run it. It can run with minimal ram storage and cpu power overall but it has its drawbacks. Some drawbacks include needing training to perform and also having accuracy and performance issues compared to LLMs. Overall I agree with the fact that TinyML can do well on a microcontroller but lacks performance. This lack of performance capabilities is because it can't use open ended reasoning and is confined to whatever it is trained on which prevents it making its "own decisions". It also cant use language but actions have to be manually coded in.
- What can a cloud-based LLM/VLM do for a robot? What are the downsides (latency, cost, connectivity dependency)?
  -LLM/VLM are powerful enough to interpret language and transform them into commands and chunks of actions. They can both handle situations which they weren't trained to do beforehand. The two main issues associated with this for my project are latency and cost. LLMs and VLMs both require a network which gives delay to the system which will reduce efficiency of the robot itself. Second the cost for running a LLM Api can cost however Ollama could possibly be a work around for that.
- Where is the interesting middle ground? Are there examples of hybrid approaches (edge inference + cloud reasoning)?
  -while researching I found 2 systems interesting. The first, required a platform called Ollama which ran LLM Ai models locally and implemented into a raspberry pi. a Raspberry Pi runs Ollama locally to handle conversational reasoning, while a camera gives input back into the system. This has no cloud dependency and is able to run on a microcontroller. The second is a hybrid setup with a minicomputer and a microcontroller paired up together in order to go around the high hardware requirements of running LLM/VLMs. This second setup would host the LLM onto the mini computer which would then be connected to a microcontroller running the servos and other components on the bot.
- Based on what you read, which approach (or combination) seems most realistic for your project given your hardware and timeline?
  -Based on my research I think using the second hybrid system consisting of a mini computer running a LLM would give us the "open ended" response we need while also fitting the cost requirements. It will also allow us to try different types of models instead of hyper focussing on perfecting our own model.

https://say-can.github.io · https://robotics-transformer2.github.io · https://code-as-policies.github.io/

# Q's
**Question 1:** does a TinyML vision model or a locally ran LLM give better accuracy in certain scenarios, and how does each approach's response latency affect task performance?
- This is interesting because it compares the efficiency trade-off of the two systems with latency issues where that means a slow response. Both models run on the same Pi, so hardware is not a changing variable. This will also is kind of obvious because it is expected that the tinyML version will be more efficient since it obviously will have a faster response

**Question 2:** On a fixed Raspberry Pi with a camera does a TinyML classifier or a locally ran VLM have better response (time, efficiency, and accuracy) and how does each handle objects outside its expected inputs reacting to unexpected scenarios.
- You can measure the outcome with this setup. Whether you completed a task, in how much time, or taking the shortest path to doing it. Running both models on the same hardware isolates the model as the only variable. The main issue with this is whether running a VLM on a Pi will hinder its maximum potential due to hardware issues..

**Question 3:** On a Raspberry Pi can you execute tasks defined by instructions in language and whether a TinyML model or a locally-run LLM can achieve higher task completion.
- This is interesting because both run on the same Pi, while keeping hardware constant. The biggest issue is that each prompt will generate a different action even if the input is the same causing it to have varied results.

**Top choice:** On a Raspberry Pi with a locally run LLM act under normal conditions and autonomous problem-solving behavior during a task when presented with these scenarios does the model recover without human intervention?

**Hypothesis:**
Under normal conditions, I predict that the locally run LLM will execute language commands given tasks reliably when the instructions are clear. Under unprecedented situations, such as new objects, contradictory instructions, or interference during task execution, the model will have trouble coping with these challenges, and its success will be dependent on whether the information it can gather from its surroundings can help it re approach the task itself. These contradictory instructions will be less of an issue for the LLM because it can be resolved with language alone but on the other hand recognizing and responding to changes in its environment and conditions requires the use of other information which the ai has to incorporate within its decision making while also following language commands. Finally, the model will show automatic recovery under stressed conditions.

# Hardware/Inventory
**Microcontrollers:**
- Arduino Uno - 32KB flash, 2KB RAM, 16MHz ATmega328P. No Wi-Fi, no Bluetooth
- Elegoo R3 - functionally identical to the Arduino Uno, same specs.

**Camera:**
- Charmed Labs Pixy2 — dedicated smart vision camera with an onboard processor that detects color signatures, lines, and barcodes at up to 60fps and Communicates with either SPI, I2C, UART, or USB. The onboard processor converts the image into a block array which gets sent to the microcontroller.

**Servos:**
- 4x 20kg torque servos — high torque
- 1x mini servo

**Wi-Fi:** None

**Power:**
- 7.4V LiPo battery pack
- UBEC

| Item | Purpose | Estimated Cost |
| :- | :- | :- |
| Raspberry Pi 5 (8GB) | Run local LLM with Ollama | ~$80 |
| MicroSD card (64GB+) | Storage | ~$10 |
| Raspberry Pi power supply (27W USB-C) | Power supply for the pi itself | ~$12 |

**Platform:** Arduino Uno and Elegoo R3 are insufficient for this project and wont be used due to its hardware limitations 2KB of RAM makes it impossible to work with and model alone running an LLMs or VLMs. The Pixy2 can actually take the burden off the LLM, which means it could serve as a layer for the LLM to receive already processed visual data. For the research question to work on a locally-run LLM responding to natural language, completing tasks, and recovering automatically from an unprecedented scenario I will need a Raspberry Pi 5 with 8GB RAM as my platform. This is the hardware that can run a small LLM with Ollama at a usable speed. The interface with your Pixy2 for visual input, and control your servos through a microcontroller and another external computer.
[image]

# Project Roadmap
| Time | deadline |
| :- | :- |
| Apr 2026 | Literature Review & Research Question Finalization |
| Late Apr 2026 | Hardware Acquisition & Basic Setup |
| Early May 2026 | EXAM BUFFER-work if possible |
| Mid May 2026 | EXAM BUFFER-work if possible |
| Late May 2026 | LLM Integration |
| Jun 2026 | Unprecedented Scenario Testing & Data Collection |
| Jul 2026 | Analysis, Writeup & Figures |
| Aug 2026 | Final Report & Presentation |
