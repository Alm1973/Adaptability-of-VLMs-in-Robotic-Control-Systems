# Early planning (April 2026) — Project Roadmap + Ideas & Open Questions

> Sources (both imported verbatim, 2026-09-28):
> 1. Google Doc "Qs " (Project Roadmap) · (private Google Drive link removed) · 2026-04-11. Same content as Roadmap.docx (private Google Drive link removed) — not imported twice.
> 2. Ideas_and_Open_Questions.docx · (private Google Drive link removed) · 2026-04-11
> Earliest framing (Raspberry Pi + Pixy2, text-only LLM, object sorting). Superseded — useful only for the "how the project evolved" story.

---
# 1. Project Roadmap
**LLM-Driven Robotics: Task Performance & Autonomous Recovery** — April 2026 – August 2026 | Living Document

## Research Question
On a fixed Raspberry Pi platform performing natural-language-commanded object sorting, how does a locally-run LLM perform under normal conditions and how does it respond to unprecedented scenarios — including novel objects, contradictory instructions, and mid-task disruptions — and does the model demonstrate meaningful recovery without human intervention?

## Timeline Overview
*Planned at 5–7 hours/week. Exam period (May) is built in as a lighter phase.*

| Phase | Timeline | Deliverable | Status |
| :- | :- | :- | :- |
| Phase 1 | Apr 2026 | Literature Review & Research Question Finalization | In Progress |
| Phase 2 | Late Apr 2026 | Hardware Acquisition & Basic Setup | Pending |
| Phase 3 | Early May 2026 | Baseline Experiment (No AI) | Not Started |
| Phase 4 | Mid May 2026 | EXAM BUFFER — Lighter Week | Planned |
| Phase 5 | Late May 2026 | LLM Integration & Basic Task Testing | Not Started |
| Phase 6 | Jun 2026 | Unprecedented Scenario Testing & Data Collection | Not Started |
| Phase 7 | Jul 2026 | Analysis, Writeup & Figures | Not Started |
| Phase 8 | Aug 2026 | Final Report & Presentation | Not Started |

## Phase Details
**Phase 1 — Literature Review & Research Question Finalization (April 2026)**
- Complete 8–10 source literature review (TinyML, LLM robotics, edge AI, hybrid approaches)
- Finalize research question and hypothesis
- Complete Assignment 1 summary document
- Seed Ideas & Open Questions document

**Phase 2 — Hardware Acquisition & Basic Setup (Late April 2026)**
- Purchase Raspberry Pi 5 (8GB), MicroSD card, USB-C power supply
- Install Raspberry Pi OS, Ollama, and a baseline LLM (Phi-3 Mini or TinyLlama)
- Wire Pixy2 camera to Pi via USB or SPI
- Connect servos via Arduino Uno as low-level motor controller (serial communication from Pi)
- Confirm end-to-end communication: Pi → Arduino → servos
- Confirm Pixy2 object detection pipeline outputs structured data to Pi

**Phase 3 — Baseline Experiment: No AI (Early May 2026)**
- Build a simple rule-based sorting system using Pixy2 color signatures only
- Run 20–30 trials, log task completion rate, time per sort, error types
- This becomes the performance floor: everything AI does will be compared to this
- Document failure cases — what does the rule-based system get wrong?

**Phase 4 — Exam Buffer (Mid May 2026)** — *Lighter week intentionally. Use this time only for background reading or light setup — no pressure to hit milestones.*

**Phase 5 — LLM Integration & Basic Task Testing (Late May 2026)**
- Integrate LLM via Ollama into sorting pipeline
- LLM receives Pixy2 scene descriptions as text input, outputs sort decisions
- Run standard sorting trials with clear natural language instructions
- Measure: task completion rate, response latency, error types
- Compare directly against Phase 3 baseline

**Phase 6 — Unprecedented Scenario Testing & Data Collection (June 2026)**
- Design 4–6 unprecedented scenario types:
  - Novel object (never seen before, no color signature trained)
  - Contradictory instruction mid-task
  - Physical disruption (object moved during sort)
  - Ambiguous instruction ('sort by which looks more valuable')
  - Object that fits no defined category
- Run 10+ trials per scenario type, score recovery using defined rubric:
  - Level 0: Freeze / error | Level 1: Acknowledges failure | Level 2: Attempts workaround | Level 3: Successful recovery
- Log all outputs verbatim for qualitative analysis

**Phase 7 — Analysis & Writeup (July 2026)**
- Compile and visualize data: completion rates, latency distributions, recovery scores by scenario type
- Write results and discussion sections
- Identify where LLM succeeded, where it failed, and whether failures follow predictable patterns
- Connect findings back to hypothesis and literature

**Phase 8 — Final Report & Presentation (August 2026)**
- Complete full research paper draft
- Build presentation slides
- Record or present demo video of robot performing task
- Submit / share final deliverable

## Key Risks & Mitigations
| Risk | Likelihood | Mitigation |
| :- | :- | :- |
| Pi 5 ships late | Low | Order immediately in Phase 2 |
| LLM too slow on Pi 5 | Medium | Test smaller models (TinyLlama, Phi-3 Mini); accept latency as a data point |
| Pixy2-Pi integration issues | Medium | Budget 1 full week for setup debugging |
| Exams compress schedule | High | Phase 4 buffer already planned; Phase 5 can slip 1 week without cascading |
| Unprecedented scenarios hard to standardize | Medium | Define rubric before running experiments, not after |

---
# 2. Ideas & Open Questions
**LLM-Driven Robotics: Task Performance & Autonomous Recovery** — Living document

## Core Research Question & Hypothesis
**Question:** On a fixed Raspberry Pi platform performing natural-language-commanded object sorting, how does a locally-run LLM perform under normal conditions and how does it respond to unprecedented scenarios — including novel objects, contradictory instructions, and mid-task disruptions — and does the model demonstrate meaningful recovery without human intervention?

**Hypothesis:** The LLM will perform reliably under normal conditions but will struggle with unprecedented scenarios. Recovery will occur in some but not all cases, and failures will follow a consistent pattern: the model produces confident, plausible-sounding responses that are physically incorrect because it cannot verify whether its actions worked. This suggests the ceiling for LLM-based robotics on constrained hardware is not raw language intelligence but perceptual feedback.

## Open Questions
**1. LLM vs. Hybrid — Which Approach for Which Sub-Task?**
❓ Should different parts of the pipeline use different AI approaches? The Pixy2 already does fast color detection onboard. The LLM handles language and reasoning. Is there a third layer needed for physical feedback? Could a hybrid where Pixy2 handles routine detection and the LLM only handles edge cases outperform pure LLM for everything?
- The LLM is best suited for: interpreting natural language instructions, handling ambiguity, reasoning about novel scenarios
- The Pixy2 is best suited for: fast, reliable color/shape detection on known objects at 60fps
- Gap to fill: physical feedback loop — how does the robot confirm its action worked?
- Open: is there a lightweight way to add a confirmation step without adding a full VLM?

**2. What Does 'Better' Mean in This Comparison?**
❓ How do we define success, and are we measuring the right things? Task completion rate is clean and countable but may miss important nuance. A robot that completes 80% of tasks but fails catastrophically on the other 20% is very different from one that fails gracefully every time.
- Primary metric: Task completion rate (binary per trial — correct sort or not)
- Secondary metric: Response latency (time from instruction to action)
- Recovery metric: 4-level rubric (0=freeze, 1=acknowledges failure, 2=attempts workaround, 3=successful recovery)
- Qualitative metric: Nature of failure — does the model fail loudly (expresses uncertainty) or silently (acts confidently but wrongly)?
- Open: should we weight graceful failure higher than silent failure in our scoring?

**3. Response Time Requirements**
❓ How fast does the robot actually need to respond for this task? For a stationary arm doing sorting, a 3-5 second LLM response may be acceptable. For navigation or real-time obstacle avoidance, it would not be. Our task choice insulates us from the worst latency problems, but we should still measure and report it.
- Sorting task: each trial is discrete, so latency per decision matters less than for continuous control
- Expected LLM response time on Pi 5: 3–15 seconds depending on model size and prompt length
- Acceptable threshold: TBD based on baseline experiment — what does the rule-based system take?
- Open: does latency affect the quality of the LLM's response, or just the speed?

**4. What Is the Simplest Task That Still Tests the Research Question?**
❓ What is the minimum viable experiment? We want to test LLM performance under normal conditions AND unprecedented scenarios. The task needs to be simple enough to run many trials but complex enough that the LLM's reasoning actually matters.
- Candidate: sort 3 object categories by color using Pixy2 + LLM, with natural language commands
- Normal condition: 'Put the red object on the left and the blue object on the right'
- Unprecedented condition 1: introduce a purple object (ambiguous color category)
- Unprecedented condition 2: 'Actually, swap those' mid-task (contradictory instruction)
- Unprecedented condition 3: physically move the object while the robot is mid-sort
- Open: how many categories and objects are needed for results to be statistically meaningful?

**5. Sending Visual Data Without a Full VLM**
❓ If the LLM can't see, how do we give it visual context cheaply? The LLM receives text only. The Pixy2 outputs structured object data (color, position, size) not raw images. This is actually useful: we can pass Pixy2's output directly to the LLM as a text scene description without needing a VLM at all.
- Pixy2 outputs: object color signature, bounding box (x, y, width, height), tracking ID
- This can be formatted as: 'Object detected: red, center position x=150 y=200, size 40x30px'
- LLM receives this as its 'vision' — structured text rather than raw pixels
- Limitation: LLM cannot see texture, shape beyond bounding box, or spatial relationships the Pixy2 doesn't report
- Open: is Pixy2's structured output sufficient for the LLM to make good sorting decisions, or does it need richer scene description?

## Yann LeCun & Physical AI — Relevant or Not?
LeCun's core argument is that LLMs are fundamentally limited for physical robotics because they predict tokens, not physical reality. They have no model of how the world works, cannot predict the consequences of their actions, and therefore cannot plan or recover from failure in a grounded way. His proposed alternative is world models — AI systems trained on video and sensor data to build internal simulations of the physical world, using the Joint Embedding Predictive Architecture (JEPA).

Relevance to this project: HIGH. LeCun's critique directly predicts what we expect to find — that the LLM will fail at unprecedented scenarios not because it lacks intelligence but because it lacks physical grounding. It produces plausible-sounding text responses but has no way to verify whether those responses correspond to real-world outcomes. This is exactly the failure mode we are testing for.
- LeCun's argument supports our hypothesis that the LLM's ceiling is perceptual feedback, not language reasoning
- It also suggests our experiment is poking at a genuinely open and important problem in AI, not just a toy comparison
- Open question: does our project inadvertently demonstrate LeCun's point, or does the LLM surprise us by recovering well through language reasoning alone?
- Worth noting: LeCun's AMI Labs raised $1.03B in March 2026 to pursue world models — this is an active frontier research area, not settled science
- His LeWorldModel (LeWM) achieved 96% success on a robotic Push-T benchmark using JEPA — far beyond what a text-only LLM could do on a physical task

## Ideas for Future Extensions
- Add a VLM layer (LLaVA) and compare recovery rates against pure LLM — does vision grounding actually help?
- Test the same pipeline with a larger LLM (7B vs 1B) — does model size change recovery behavior?
- Add a simple confirmation loop: after acting, LLM is told 'the object is still in place' and asked to try again
- Explore prompt engineering: does chain-of-thought prompting improve recovery in unprecedented scenarios?
- Connect findings to LeCun's world model framework: could a lightweight world model replace the LLM for this task?

## Sources to Follow Up
- LeCun, Yann. 'A Path Towards Autonomous Machine Intelligence.' 2022. — foundational paper on JEPA and world models
- Google DeepMind AutoRT (2024) — real-world multi-robot LLM control
- Sikorski et al. 'Deployment of NLP and LLM Techniques to Control Mobile Robots at the Edge' (2024) — GPT-4 vs local LLaMA 2 for robot control
- SiliconWit TinyML benchmark (2025) — concrete Pi performance data for LLM inference
- MIT Technology Review: 'Yann LeCun's new venture is a contrarian bet against large language models' (Jan 2026)
