# Assignment 2 — Lock down the research question; physical AI & control theory (April–May 2026)

> Source: Google Doc "Assigment2" · (private Google Drive link removed)
> Created 2026-04-24 · Last modified 2026-05-05 · Imported 2026-09-28 (verbatim)
> Contains the mentor's assignment text (instructions) followed by Shaurya's answers. Includes the first formal RQ, metrics (30-run plan), falsifiable hypothesis, and top-3 risks.

Assignment 1: Lock Down the Research Question
Priority: High | Deadline: Before Session 3

This is the one that matters. We've now spent two sessions circling your topic: TinyML vs. LLM, then cloud vs. local, now open-ended reasoning and physical AI. That's fine for exploration, but it's time to converge.

As I said in the session, a good research question should have three things:
- A problem: Something concrete that needs solving, not just an area you're interested in
- A proposed approach/solution: Your angle on how to tackle it (which AI, on what hardware, doing what task).
- A direction: What does success look like, and what would you measure to know you got there?

(a) Write one leading research question. Not three this time, just one. It should fit in 2–3 sentences and satisfy all three criteria above. Your current top-choice question ("does the model recover without human intervention") is getting closer but is still fuzzy: what counts as "recovery"? What's the task? What's the failure mode you're injecting? Tighten it.

(b) Immediately below the question, write a measurable outcome. Pick one or two metrics. Examples: task completion rate under X conditions vs. Y conditions; time-to-recovery in seconds; number of human interventions needed per 10 trials; success rate on a fixed set of unexpected scenarios. If you cannot write down what you would measure, the question isn't sharp enough yet.

(c) Write a revised one-paragraph hypothesis that reflects the shift we talked about, which is away from TinyML vs. LLM comparison and toward autonomous / recovery. Be specific about what you think will happen AND what you think will break. The hypothesis should be falsifiable: if you ran the experiment, there should be a clear outcome that would prove you wrong.

(d) Under your hypothesis, list the 3 biggest risks or unknowns that could make this project not work. Be honest and transparent. (Example: "If Qwen-3B is too slow on Pi 5 to react in time, the recovery behavior becomes untestable.")

## Research question:
Can a locally run large language model on a Raspberry Pi 5 autonomously recover from task disrupting situations such as a, including occlusion, changes in its environment, and abnormal targets, by allowing dynamic reasoning and adaptive camera repositioning strategies. The study will push a fixed set of failure factors into a defined object verification task, and measure whether the model detects the failure, adapts its plan, and ultimately completes the task on its own.

## Measurable Outcomes:
**Primary metric:** The completion percentage (%) in 10 runs each for scenarios of occlusions, environment changes (lighting change, blurring of camera, or movement of an object midway through the task), and an abnormal/unexpected target, totaling 30 runs or more if needed.

**Secondary metric:** The number of human interactions needed per 10 runs each for these scenarios. An autonomous recovery is considered complete when the number of human interactions in a run equals zero. Every manual correction equals one failure run and points against the score.

## (c) Revised Hypothesis
With proper visibility and known target, the locally supervised LLM should accomplish its object identification objective with relative ease, with likely success in 8 or more out of 10 trials. Introducing failure cases will cause varying levels of disruption: occlusions will be the easiest to overcome, since the model will have a tangible action to execute (moving its camera to another position) to correct the issue, with successful completion of the task indicated by a visual cue (object becoming visible once more). Environmental disruptions – like sudden changes in light and/or relocation of the target object will moderately hinder performance, as the LLM must first recognize that its assumptions about the environment are outdated before being able to plan a new course of action, which smaller models tend to do poorly at. Anomalies will be the most difficult type of failure to recover from autonomously, as the model will not have sufficient classification terminology to determine the identity of the object and will likely make an incorrect judgment call or get stuck in a recursive thinking loop. The hypothesis is falsifiable, as if the model achieves autonomous recovery over 70% across all three types of failures, it would be incorrect; however, if recovery is maximized during occlusion and least during anomaly identification it would be correct.

## (d) Three Biggest Risks and Unknowns
**1. Inference latency may make dynamic reasoning non-functional in practice.** Model inference latency might render dynamic reasoning useless. The time spent on inference by the model might be as long as tens of seconds per cycle; in such a case, the camera repositioning will not be based on feedback but rather represent an open loop approximation. If latency is long, the model will try different camera positions but will not get a response in time, and by the time it receives a reply, the context may become outdated anyway.

**2. Failure injection may not be reproducible enough to be scientifically valid.** Failure injections are unlikely to be reproducible enough for a valid scientific experiment. Occlusion depth, lighting change extent, and how much an abnormal object differs from typical targets can hardly be discrete variables. You will treat all of them as some constants of sorts. However, if in Trial 3 the object is covered in a more pronounced way compared to Trial 7 just because you shifted the blocking object by several centimeters during the setup phase, you will not be conducting a reproducible test. You must strictly specify the location of the blocking objects, light placement, and a set of abnormal objects used before the test.

**3. The camera repositioning action space may be too underspecified for the model to use reliably.** The space of possible actions in terms of camera repositioning might be too underspecified for the model to utilize them correctly. To be able to conduct camera repositioning experiments, the model needs to know what kind of actions are available, how it can instruct the camera to perform them, and what the criteria of success are. Unless there is a strict specification of all this information in the prompt or system context, the model will generate action prompts that are too broad ("slightly move to the left," "rotate to this side"), inconsistent between the trials, or simply impossible to perform physically.

Assignment 2: Physical AI & Control Theory
Priority: High | Deadline: Before Session 3

You raised physical AI (Yann LeCun) and I raised control theory. Both are relevant to your new direction, and you should understand enough about them to decide whether either changes your approach.

(a) Find and read/watch at least 2 sources on Yann LeCun's physical AI / world models concept. A good starting point is his Meta/FAIR talks and the **JEPA (Joint Embedding Predictive Architecture)** papers. Look specifically for: is there any open-source code or working prototype? Or is this still theoretical? Log findings in your literature review with proper citations.

(b) Find and read/watch at least 1 source on control theory basics as applied to robotics. You don't need to master Bellman equations, just understand the framing. Boston Dynamics material is fine as an entry point, but try to find something that actually explains the optimization framing (cost functions, constraints). Log it.

(c) Read at least 2 of the three papers/projects I sent in chat: SayCan, RT-2, and Code as Policies. These are the "LLM-as-robot-brain" genre and directly relevant to your new question. Log them with 2–3 sentence summaries each.

(d) Add a short section to your literature review doc titled "Where my project fits." In 1–2 paragraphs, position your project relative to these three camps: (1) control theory / classical robotics, (2) LLM-as-planner (SayCan, RT-2, Code as Policies), (3) physical AI / world models. Which camp is your project in? Which one is it borrowing from?

# (A) Yann LeCun's Physical AI & JEPA World Models
**Citation:** Assran, M., et al. (2023). "I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture." Meta AI Blog. Official codebase: https://github.com/facebookresearch/ijepa

I-JEPA stands for Image Joint Embedding Predictive Architecture and is the first implemented version of LeCun's theoretical proposal. Instead of generating the reconstruction of missing pixels I-JEPA learns through prediction of the latent representation of the missing region based on context representations, which ensures semantic understanding of the world and avoids learning low level image details. The computational efficiency of the method is notable because it does not need handcrafted data augmentation or contrastive pairs, yet it shows strong transfer learning capabilities even when using a pre-trained frozen backbone. The official implementation is available on GitHub (facebookresearch/ijepa).

Key take away: The above explanation might be seen as theoretical because there is no experimental validation. However, it should be noted that the method described above has been implemented into an actual model (I-JEPA), which was trained on several image datasets including ImageNet and publicly released on GitHub with source code.

**Citation:** Bardes, A., et al. (2025). "V-JEPA 2: Self-supervised video world models for understanding, prediction, and planning." Meta AI. Open-source: https://github.com/facebookresearch/vjepa2

The second version of V-JEPA expands the framework to video at internet scale pretrained on more than one million hours of video and one million images via a masked spatio-temporal patch prediction objective. The novelty compared to the original design is V-JEPA 2-AC, which is a variant trained on only 62 hours of unlabeled robot video data from the Droid dataset. The network learns to predict future video embeddings given robot actions, allowing zero-shot planning using model-predictive control (MPC).

When tested in a practical environment, V-JEPA 2-AC was applied to a Franka robot arm and achieved the same tasks in novel environments without any specific task training, reward signals, and calibration for each setup. In this approach, the planning process consists of minimizing the difference between the imagined future states and a target image by employing the Cross-Entropy Method (CEM). One planning step takes roughly 16 seconds.

Key takeaway regarding open-source and theoretical aspects: V-JEPA 2 is completely open-source (GitHub, HuggingFace). It features functional robotic experiments on real hardware. However, the experiments were carried out with high-end resources (Franka arm, GPU servers) impossible to emulate in a Raspberry Pi setup. In your project, JEPA serves as an inspiring architecture but not a practical framework — it is important to focus on the concept of representation prediction rather than pixel-based reconstruction.

## Summary:
Key point: LeCun argues that existing LLMs are unable to model physical reality due to their reliance on the processing of language tokens instead of sensory experience. The idea of JEPA is to train the AI system through its perception of the real world (e.g., video feed or sensors) to predict future states of the environment using an abstract representation. For a robot, it implies predicting the outcome of taking certain actions in advance. As we have mentioned above, this concept is entirely different from LLMs being a planner.

# (B) Control Theory Basics Applied to Robotics
**Citation:** Robotics Meta. (2026). "Introduction to Optimal Control Theory." https://roboticsmeta.com/introduction-to-optimal-control-theory/

The control theory gives the mathematics to get your robot to behave the way you want it to reliably in real-world conditions. The key framing is as follows: given the mathematical model of the movements of your robot (dynamics), find the control policy (the rule for taking actions) that will optimize some given cost function within certain constraints. Mathematical model (dynamics): a series of equations that describe how the robot evolves in response to the inputs. In case of the servo-driven robot arm, this is joint positions and their speeds, the applied torque, and the physical boundaries. Cost function (performance index): the mathematical representation of what is meant by "good performance." Examples include minimal time to target, minimal energy consumption, minimal distance from the intended path, or minimal jerk (smoothness of movement). The choice of the cost function is responsible for determining what the robot should optimize. Constraints: hard boundaries which the robot can never exceed – maximum torque of the servos, the physical boundaries of the joints, boundaries of collision-free space, voltage on the battery, etc. LQR (Linear Quadratic Regulator): assumes the dynamics are described via linear equations and the cost function is quadratic; provides a closed-form solution. Popular for controlling the hovering position of drones and for biped balancing. MPC (Model Predictive Control): more advanced and widespread approach, based on planning the next steps several steps into the future. The robot considers only a short horizon into the future but solves the optimization problem for that period. It allows the robot to plan its path while obeying the specified constraints – for instance, "don't bump into the wall."
