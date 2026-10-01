# Literature Review — Physical AI, Control Theory & LLM-as-Robot-Brain (Assignment 2, Session 3 prep)

> Source: Google Doc "Assignment2_LitReview" · (private Google Drive link removed)
> Created 2026-04-27 · Imported 2026-09-28 (verbatim)
> Note: written during the Raspberry Pi / Pixy2 phase; "Where my project fits" needs updating for the final YOLO-World + Qwen2.5-VL system. Mentor flagged (status report, Jul 24) that lit review should be redone in own words on papers actually read.

**Literature Review**
*Assignment 2: Physical AI, Control Theory & LLM-as-Robot-Brain*
Session 3 Preparation | Raspberry Pi Robot Project

# A. Yann LeCun's Physical AI & JEPA World Models
## Source 1: I-JEPA — Meta AI Blog (Official Release)
**Citation:** Assran, M., et al. (2023). "I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture." Meta AI Blog. Official codebase: https://github.com/facebookresearch/ijepa

The Image Joint Embedding Predictive Architecture (I-JEPA) is the first working prototype of LeCun's theoretical framework. Rather than predicting pixel-level reconstructions, I-JEPA learns by predicting the abstract representation of missing image regions from the representations of surrounding context — operating entirely in latent space. This forces the model to develop semantically meaningful internal representations of the world rather than surface-level texture memorization.

The model is computationally efficient — it does not require hand-crafted data augmentations or contrastive pairs — and demonstrates strong transfer learning performance across downstream vision tasks with a frozen backbone. The official PyTorch codebase is open-source on GitHub (facebookresearch/ijepa) and is actively maintained by Meta FAIR.

Key finding: This is NOT purely theoretical. I-JEPA is a working, open-source model that has been trained, evaluated on benchmarks like ImageNet, and released with full training code. It represents the first concrete implementation of LeCun's proposed architecture.

## Source 2: V-JEPA 2 — Meta AI (2025)
**Citation:** Bardes, A., et al. (2025). "V-JEPA 2: Self-supervised video world models for understanding, prediction, and planning." Meta AI. Open-source: https://github.com/facebookresearch/vjepa2

V-JEPA 2 extends the architecture to video at internet scale — pretrained on over one million hours of video combined with one million images, using a masked spatio-temporal patch prediction objective. The key innovation over its predecessor is V-JEPA 2-AC, an action-conditioned variant fine-tuned on only 62 hours of unlabeled robot video from the Droid dataset. This variant learns to predict future video embeddings conditioned on robot actions, enabling zero-shot planning through model-predictive control (MPC).

In practice, V-JEPA 2-AC was deployed on a Franka robot arm in new, unseen environments and successfully performed reaching, grasping, and pick-and-place tasks — without task-specific training, reward supervision, or calibration for each new setting. Planning worked by minimizing the distance between imagined future states and a visual goal image, using the Cross-Entropy Method (CEM). Planning takes approximately 16 seconds per step.

Key finding re: open-source vs. theoretical: V-JEPA 2 is fully open-source (GitHub, HuggingFace). It has working robotics prototypes demonstrated on real hardware. However, the robot demonstrations use large-scale infrastructure (Franka arm, GPU servers) far beyond what a Raspberry Pi project can replicate. For your project, JEPA is an inspirational architectural concept rather than a deployable tool — what matters is the underlying idea that prediction in representation space is more useful than pixel-level reconstruction.

## Conceptual Summary: What JEPA Means for Robotics
LeCun's central argument is that current LLMs lack a grounded model of physical reality because they process language tokens rather than sensory experience. JEPA proposes to fix this by training AI on observations of the world (video, sensor data) and asking it to predict future states in abstract space — building an internal world model. For a robot, this means the AI can imagine the consequences of its actions before executing them. This is fundamentally different from the LLM-as-planner approach (Section C), where the model has no internal model of physics at all and relies entirely on language-encoded knowledge.

# B. Control Theory Basics Applied to Robotics
## Source: Robotics Meta — "Introduction to Optimal Control Theory" (2026)
**Citation:** Robotics Meta. (2026). "Introduction to Optimal Control Theory." https://roboticsmeta.com/introduction-to-optimal-control-theory/

Control theory provides the mathematical framework for making a robot do what you want it to do, reliably, under real-world conditions. The core framing is: given a mathematical model of how your robot moves (dynamics), find a control policy (a rule for choosing actions) that minimizes a cost function subject to constraints.

### The Three Core Elements
- Mathematical Model (Dynamics): A set of equations describing how the robot's state changes in response to inputs. For a servo-driven robot arm, this includes joint positions, velocities, torques, and physical limits.
- Cost Function (Performance Index): A mathematical expression that quantifies what 'good' looks like. Examples: minimize time-to-target, minimize energy consumption, minimize deviation from a planned path, or minimize jerk (smoothness). The choice of cost function entirely determines what behavior the robot optimizes for.
- Constraints: Hard limits the robot cannot violate — servo torque limits, physical joint ranges, collision boundaries, battery voltage thresholds.

### Key Control Methods
LQR (Linear Quadratic Regulator) is the classical starting point: it assumes linear dynamics and a quadratic cost function, which allows an analytical (closed-form) solution. It is widely used for drone hovering and biped balance. MPC (Model Predictive Control) is more powerful and more common in modern robotics: rather than solving the entire trajectory at once, the robot looks ahead a short horizon, plans an optimal path for that window, executes the first step, and repeats. This allows the robot to handle constraints explicitly — for example, 'do not hit the wall' — and re-plan as the environment changes.

Why this matters for your project: Your project does not use classical control theory directly. You are not hand-designing a cost function for servo movements. However, understanding control theory gives you the vocabulary to describe what the LLM is actually doing when it generates commands: it is implicitly specifying actions without explicitly optimizing a cost function. This is both the power and the limitation of the LLM-as-planner approach — LLMs can reason about goals in language but cannot guarantee physical optimality or safety in the way a cost-constrained controller can.

Additional reference (MIT Underactuated Robotics Course): Tedrake, R. (2024). "Trajectory Optimization." MIT 6.832: Underactuated Robotics. https://underactuated.mit.edu/trajopt.html — This provides the formal optimization framing, including direct transcription, shooting methods, and augmented Lagrangian approaches used in real robotics research.

# C. LLM-as-Robot-Brain: SayCan, RT-2, and Code as Policies
## Paper 1: SayCan — "Do As I Can, Not As I Say"
**Citation:** Ahn, M., et al. (2022). "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." Google Robotics. arXiv:2204.01691. https://say-can.github.io

SayCan addresses a fundamental problem: LLMs have rich world knowledge but no grounding in what a specific robot can actually do in a specific environment. The solution is to combine the LLM's semantic reasoning with affordance functions — value functions trained via reinforcement learning that estimate the probability that a given skill will succeed given the current environment state. At each planning step, SayCan multiplies the LLM's probability of a skill being semantically appropriate by the skill's affordance score, selecting the action that is both linguistically plausible and physically feasible.

Results: On 101 household tasks, SayCan with PaLM achieved 84% correct action sequence selection and 74% execution success. The system successfully handled multi-step tasks (up to 8 steps). Open-source simulation code is available on GitHub. The key insight is that the affordance layer 'overrides' the LLM when a linguistically reasonable action is physically impossible — the robot does not try to pick up an object it cannot reach.

Relevance to your project: SayCan is architecturally similar to what you are building. You have an LLM (via Ollama on Raspberry Pi) that plans actions, and the Pixy2 camera provides environmental state that could function as a simplified affordance layer — if the camera doesn't see the target object, the LLM's plan to interact with it should be overridden. The key difference is that SayCan uses pre-trained RL-based skill primitives, while your project will likely use simpler servo commands as primitives.

## Paper 2: RT-2 — Vision-Language-Action Models
**Citation:** Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." Google DeepMind. arXiv:2307.15818. https://robotics-transformer2.github.io

RT-2 takes a different approach from SayCan: instead of having a separate LLM and skill library, it fine-tunes a large vision-language model (VLM) end-to-end on both web data and robot demonstration data simultaneously. Robot actions are represented as text tokens (discretized motor commands), so the same model that processes language and images also directly outputs motor commands. This is called a Vision-Language-Action (VLA) model.

The key result is emergent generalization: RT-2 successfully performs tasks with objects, relationships, and phrasings it was never explicitly trained on in robot data — because it draws on web-scale knowledge baked into the VLM backbone. With chain-of-thought prompting, it can perform multi-stage semantic reasoning (e.g., inferring that a rock could serve as an improvised hammer). The model shows that improvements in language model quality directly translate to improvements in robotic performance.

Relevance to your project: RT-2 represents the far end of the scale from your project — it requires 10B+ parameter models and large-scale robot training data. However, the insight is directly relevant: the reason you are running an LLM locally on a Raspberry Pi is precisely to access web-scale reasoning for open-ended task performance, which is the same motivation as RT-2. Your project is a resource-constrained version of the same core idea.

## Paper 3: Code as Policies — LLM Programs for Embodied Control
**Citation:** Liang, J., et al. (2022/2023). "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023. arXiv:2209.07753. https://code-as-policies.github.io

Code as Policies proposes using code-writing LLMs to generate executable Python robot policy code from natural language commands, rather than planning in natural language steps. Given a few example (command, code) pairs via few-shot prompting, the LLM synthesizes new control programs that call existing perception and motor APIs. Because the output is executable code rather than language, it can express feedback loops, conditional logic, geometric reasoning via libraries like NumPy and Shapely, and precise numerical values — capabilities that pure language planning lacks.

The system uses hierarchical code generation: when the LLM needs a capability it doesn't have defined, it recursively defines the function. This allows arbitrarily complex behaviors to emerge from a simple prompt interface. The approach is modality-agnostic and has been demonstrated on tabletop manipulation and mobile robot platforms.

Relevance to your project: This is the approach most immediately applicable to your setup. Instead of asking Ollama to output a sequence of English steps, you could prompt it to write Python functions that call your servo control API directly. This would give your LLM spatial-geometric reasoning it currently lacks and allow it to express conditional recovery behaviors explicitly in code rather than relying on the LLM to re-plan in prose.

# D. Where My Project Fits
My project sits most squarely in the LLM-as-planner camp — the same genre as SayCan, RT-2, and Code as Policies. The core question I am investigating is whether a locally run LLM on a Raspberry Pi can interpret natural language commands, execute tasks with servo-driven hardware, and recover autonomously from unexpected situations. This is structurally identical to what SayCan and Code as Policies do, compressed to consumer hardware: the LLM is the cognitive layer, the Pixy2 provides visual perception, and servo primitives are the action space. The key difference from SayCan is that I do not have pre-trained affordance value functions — my version of 'grounding' will come from the Pixy2's color-block output as a simplified state representation. The key difference from RT-2 is scale: I cannot fine-tune a 10B parameter model on robot data, so I rely on Ollama's quantized models for zero-shot generalization from web-scale pretraining alone.

My project borrows from control theory in a limited but important way: the low-level servo control is essentially classical feedback control (the servo's internal PID loop), and understanding cost functions helps me reason about what my evaluation criteria actually are — task completion rate, recovery success, and latency are all informal cost-function proxies. What my project does not borrow is the formal optimization framing of MPC or LQR; the LLM generates motor commands without explicitly minimizing a cost function, which is both its flexibility and its weakness. My project is entirely separate from the physical AI / world models camp: I am not training any model on sensory data, and the LLM I use has no internal model of physics — it reasons about the world through language alone. JEPA-style world models represent a future trajectory for this kind of work, where the robot's planning system would have genuine predictive physical understanding rather than purely linguistic reasoning. For now, my project tests the boundary of how far language-based reasoning can take a resource-constrained physical system.

# References
Ahn, M., Brohan, A., Brown, N., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691. https://say-can.github.io

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. CVPR 2023. https://github.com/facebookresearch/ijepa

Bardes, A., Garrido, Q., Mur-Labadia, L., et al. (2025). V-JEPA 2: Self-supervised video world models for understanding, prediction, and planning. Meta AI. https://github.com/facebookresearch/vjepa2

Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. Google DeepMind. arXiv:2307.15818. https://robotics-transformer2.github.io

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., & Zeng, A. (2023). Code as Policies: Language Model Programs for Embodied Control. ICRA 2023. arXiv:2209.07753. https://code-as-policies.github.io

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Meta AI / FAIR position paper.

Robotics Meta. (2026). Introduction to Optimal Control Theory. https://roboticsmeta.com/introduction-to-optimal-control-theory/

Tedrake, R. (2024). Trajectory Optimization. MIT 6.832: Underactuated Robotics (Spring 2024). https://underactuated.mit.edu/trajopt.html
