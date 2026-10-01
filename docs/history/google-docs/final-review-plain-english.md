# AVI — Final Review (plain-English version)

> Source: Google Doc " Final Review " · (private Google Drive link removed)
> Created 2026-09-11 · Last modified 2026-09-11 · Imported 2026-09-28 (verbatim except: the 48-trial table in Part 4 is not repeated — every number in it is identical to the table in final-review-technical.md; the per-row "In plain words" column is summarised below)
> Code link given in doc: github.com/Alm1973/avi-occlusion-recovery

Shaurya Khidake · Experiment run on 1 September 2026 · Report written 11 September 2026

# PART 1 — What is this project?
## The one-sentence version
I built a small robot camera that watches a red cup, and I tested whether it can be *fooled* — and whether adding an AI that "looks and thinks" makes it harder to fool.

## The three parts of the software, and what each one does
This is the most important thing to understand, because the whole experiment is about figuring out which of these three parts deserves the credit.

| Part | What it actually is | Think of it as... | How often it runs | How long it takes |
| :- | :- | :- | :- | :- |
| **The detector** (software called YOLO-World) | A program you give a phrase to — I gave it "red cup" — and it draws a box around anything in the picture it thinks matches. It also gives a **confidence score** from 0 to 1, meaning "how sure am I". | A fast but slightly careless lookout who glances at every single frame and shouts "I see it, over there!" | **Every frame** (~28 times a second) | About 14 milliseconds. A millisecond (ms) is one thousandth of a second. |
| **The geometry checker** (a library called OpenCV) | Plain maths on the image. Measures where things are, how big they are, whether a box overlaps another box. No opinions, no AI — just measurement. | A ruler and a protractor. | Only when the cup goes missing | About 12 ms |
| **The VLM** (Vision-Language Model) | An AI you can show a picture to and ask a question in plain English, like "is there a red cup in this image? yes or no". It's the same kind of AI as ChatGPT, except it can see pictures. Mine runs locally on my own graphics card. | A slow, careful expert you only phone up when the lookout seems confused. | **Only when something has gone wrong** — it ran on just 1.6% of frames | About 333 ms — roughly **24 times slower** than the detector |

**Why the speed difference matters:** if I asked the VLM about every single frame, the robot would run at about 3 frames per second instead of 28. It would be unusably slow. So the whole design is built around *only waking the VLM when it's actually needed*. That's called **event-gating** — a gate that only opens when an event happens.

## "Belief" — the key idea
At every moment the robot holds an opinion about the cup. I call this its **belief**. It can be:
- **CONFIRMED** — "I can see the cup, it's right there."
- **OCCLUDED** — "Something is in the way, but I think it's still behind there." (**Occluded** just means *blocked from view*. A hand in front of the cup occludes it.)
- **DISPLACED** — "It moved."
- **MISSING** — "It's gone."
- **AMBIGUOUS** — "I genuinely don't know."

## The one mistake that matters most
There are two ways the robot can be wrong, and they are **not equally bad**:

| Mistake | What it means | How bad? |
| :- | :- | :- |
| **Losing track** ("lost while present") | The cup IS there, but the robot says "I can't see it." | Annoying but **safe**. The robot knows it doesn't know, so it stops and asks for help. |
| **False belief** (the robot *lying*) | The cup is GONE, but the robot says "it's right there." | **Dangerous.** A real robot would reach out and grab empty air — or grab the wrong object. |

So I made **false belief** the main thing I measure. And I made the rule brutal: **any** false belief at all, even one frame, counts as a failed trial — no matter how good everything else was.

---
# PART 2 — The question I was trying to answer
**Original question:** Can a locally hosted vision-language model, embedded in a reasoning pipeline that uses OpenCV for spatial reasoning, recover from disruptions to object verification caused by occlusion, environment change, and unexpected objects?

**Same question, in plain English:** If I mess with the scene — cover the cup, take it away, swap it for something else, turn the lights off, move the camera — does the robot figure out what really happened? And which of my three parts is the one doing the figuring out?

That last bit is the part that makes it a real experiment rather than a demo. Anyone can build a thing that works. The scientific question is **which component is responsible**.

---
# PART 3 — How the test worked
## Each trial had three phases, 6 seconds each
1. **Baseline** — cup sitting on the marked spot, nothing happening. Let the robot settle.
2. **Disrupt** — I do the thing (cover it, move it, etc.).
3. **Recover** — I undo it, or complete the trick.

The computer told me exactly what to do at each phase and I did it by hand while it recorded every frame.

## The eight things I staged
| Name | What I physically did | Is the cup there at the end? | What it tests |
| :- | :- | :- | :- |
| **control** | Nothing. Sat still. | Yes | Does it stay calm when nothing happens? (A sanity check — if it fails this, the whole thing is broken.) |
| **occlude_hand** | Covered the cup completely with my hand, then took my hand away. | Yes | Can it hold on to the idea that the cup is still there while it can't see it? |
| **occlude_object** | Stood a cardboard box in front of the cup, then removed it. | Yes | Same, but with an object instead of a hand. |
| **removal** | Covered the cup with my hand, took the cup away under cover, then removed my hand. Empty spot. | **No** | **The sneaky one.** Can I trick it into thinking the cup is still there? |
| **substitution** | Same, but I left a pair of pink pliers on the spot instead. | **No** | Does an unexpected object fool it? |
| **identity_swap** | Same, but I swapped the red cup for a *different-coloured* cup. | **No** | The hardest one. It's still a cup — just the wrong one. Does it notice? |
| **lighting** | Turned the lamp off, then back on. Never touched the cup. | Yes | Does a change in the environment confuse it? |
| **camera_pose** | The arm panned itself sideways, then panned back. Never touched the cup. | Yes | Does it get confused when *it* is the thing that moved? |

## The second variable: clutter
I ran every scenario twice — once on a **bare** desk (just the red cup, nothing else), and once with **clutter**: six to eight other objects crowded around, **including at least two other cups or mugs in different colours**. Same object set all session.

That detail — the other cups — turned out to be the single most important thing in the whole experiment. More on that in Part 5.

## The maths: 8 × 3 × 2 = 48
8 scenarios × 3 repeats each × 2 clutter levels = **48 trials**. Took about an hour including setting up the desk between trials.

## Four things I did to stop myself cheating
This is the part that makes it science instead of a YouTube demo, so it's worth explaining properly.

| The technique | What it means | Why it matters |
| :- | :- | :- |
| **Pre-registration** | I wrote down what counts as pass/fail **in the code, before running anything**. | Stops me from looking at the results and then deciding where the bar should be. That's called "moving the goalposts" and it's how a lot of bad science happens — usually by accident. |
| **Ground truth recorded at capture time** | "Ground truth" = what was *actually* true. The computer wrote down "the cup is present / absent" from the instruction it gave me, **at the moment of recording** — not by me watching the video afterwards and deciding. | If I graded it afterwards I'd unconsciously be kinder to frames where the robot got it right. This makes that impossible. |
| **Counterbalancing** | The order of scenarios was shuffled differently in each round, using a fixed random seed (seed = 20260901) so it can be reproduced exactly. | If I always did them in the same order, "I got better at staging it by trial 40" would look like a real result. Shuffling breaks that. |
| **Edge frames excluded** | The 1 second either side of a phase change is thrown out of the grading. That's the moment my hand is mid-air and the truth is genuinely fuzzy. | Grading a frame where even a human couldn't say what's true just adds noise. These frames still get fed through the system (so the robot's thinking stays continuous) — they just don't count towards the score. 4 of every 6 seconds is graded. |

## The scoring rules (decided in advance)
| Verdict | Rule | In plain words |
| :- | :- | :- |
| **PASS** | accuracy ≥ 0.7 AND zero false belief | Got it right most of the time and never lied. |
| **PARTIAL** | accuracy ≥ 0.4 AND zero false belief | Got confused a lot, but was honest about being confused. |
| **FAIL** | anything else, or **any** false belief | Lied at least once. Automatic fail. |

**"Accuracy"** here means: during the disrupt phase, what fraction of graded frames did the robot's belief match reality? 1.0 = perfect, 0.0 = wrong on every frame.

---
# PART 4 — All 48 trials
(Full per-trial numbers: see final-review-technical.md — identical data. Column meanings from this doc: "Frames it LIED" = false-belief frames, the number that matters; at ~28 fps, 40 frames ≈ 1.4 s of confidently saying something false. "Frames it lost track" = lost-while-present, the safe kind of wrong. "How often the detector saw a cup that wasn't there" = detector false-positive rate on absent frames. "Frames it would have lied WITHOUT the VLM" = same trial replayed with the VLM off; comparing it to "Frames it LIED" is the entire point of the experiment.)

Per-row plain-words summaries used in the doc: PASS rows = "Perfect. Never fooled, never confused." (or "Good. Briefly unsure (N frames) but never lied."); PARTIAL rows = "Lost track for N frames while the cup was hidden — but honestly said 'I can't see it' the whole time. Safe, just blind."; FAIL rows = "LIED for N frames: said the cup was there after it was gone. Would have been M without the VLM." Trials 31 and 46 also "falsely 'found' it again."

## The table in one paragraph
**25 passed, 13 were confused-but-honest, 10 lied.** Look at the "Junk in frame?" column next to the red rows: **7 of the 10 failures happened with clutter on the desk.** And look at the two right-hand columns together — every single row where the VLM made a difference is a row where the detector was seeing a cup that wasn't there. That's the whole result, and the next part explains why.

---
# PART 5 — What the numbers actually say
## First, what is an "ablation"?
An **ablation** is when you remove one ingredient and re-run everything to see what that ingredient was doing. Like baking the same cake twice, once without the baking powder, to prove the baking powder is what makes it rise.

Crucially: **I did not re-stage anything.** I replayed the exact same 8,428 recorded frames through the software with the VLM switched off, and graded them with the exact same scoring code. So any difference is caused by the VLM and nothing else — not by me covering the cup slightly differently the second time.

## The headline comparison
| Version | Frames it LIED | Frames it lost track | Average accuracy | Times it wrongly "re-found" a cup that was gone |
| :- | :- | :- | :- | :- |
| **Full system** (VLM on) | **196** | 711 | 0.640 | 2 |
| **VLM removed** | **297** | 657 | 0.653 | 11 |

So turning the VLM on removed **101 frames of lying** — about 3.6 seconds' worth — and cut false "re-finds" from 11 down to 2.

## But here's the honest catch, and it's important
**42 of the 48 trials were EXACTLY the same with and without the VLM.** Not similar — identical. Only 6 trials differed at all, and just two of them account for 63 of the 101 frames (62%).

| Trial | Clutter? | Lied WITH the VLM | Lied WITHOUT it | Frames saved |
| :- | :- | :- | :- | :- |
| 9 — identity_swap | YES | 0 | 26 | 26 |
| 12 — substitution | YES | 6 | 9 | 3 |
| 14 — removal | YES | 0 | 37 | 37 |
| 29 — removal | YES | 15 | 36 | 21 |
| 30 — identity_swap | YES | 19 | 22 | 3 |
| 45 — identity_swap | YES | 17 | 28 | 11 |

So "101 frames saved" is **not** a reliable average effect — it's six specific episodes, mostly two of them. What I *can* honestly claim is about the **direction**: in all 6 trials that differed, the VLM helped, and it never once made things worse.

The statistical test for that is a **sign test** — it ignores *how much* each trial changed and only asks "which way did it go?" Getting 6 out of 6 in the same direction by pure luck is like flipping a fair coin six times and getting six heads. The odds are **p = 0.016**, or about 1 in 64. ("p" is the probability of seeing a result this lopsided if the thing you're testing did nothing at all. Below 0.05 is the usual bar for "probably not luck.")

That's a real result. It's just a **smaller** result than "the VLM cut lying by a third," which is what the raw 196-vs-297 number makes it sound like.

## Now the part that explains everything: look at the clutter column
**Every single one of those 6 trials had clutter on the desk.** That is not a coincidence. Split the results by clutter instead of by scenario and it becomes obvious:

| Desk | Trials | Lied WITH the VLM | Lied WITHOUT it | Difference |
| :- | :- | :- | :- | :- |
| **Bare** (just the red cup) | 24 | 19 | 19 | **ZERO** |
| **Cluttered** (other cups present) | 24 | 177 | 278 | **101** |

**On a bare desk, the VLM made literally no difference across 24 trials.** Not a small difference — none at all. Every single frame came out identical.

## Why? Because the detector was hallucinating
Remember the detector is the fast lookout I gave the phrase "red cup" to. I checked how often it shouted "I see it!" on frames where the red cup was **genuinely gone**:

| Desk | Frames where the cup was truly gone | Detector claimed it saw one anyway | Rate |
| :- | :- | :- | :- |
| Bare | 354 | 2 | **0.6%** — basically never wrong |
| Cluttered | 337 | 211 | **62.6%** — wrong most of the time |

Broken down further, it gets worse:

| Situation | How often the detector saw a cup that wasn't there |
| :- | :- |
| removal, cluttered desk | 103 out of 113 frames = **91%** |
| substitution, cluttered desk | 83 out of 117 = **71%** |
| identity_swap, cluttered desk | 25 out of 107 = 23% |
| removal, bare desk | 1 out of 117 = **0.9%** |
| substitution, bare desk | 1 out of 117 = **0.9%** |
| identity_swap, bare desk | 0 out of 120 = **0%** |

And it wasn't hesitating. Its average confidence on those wrong answers was **0.38**, ranging up to 0.82. In other words: **it was confidently looking at a blue mug and calling it a red cup.**

## So here is what actually happened, start to finish
1. I put other cups on the desk.
2. The fast detector started mistaking those other cups for the red one.
3. Because the detector said "I see it," the system was about to believe the red cup was still there when it wasn't — a false belief, the dangerous mistake.
4. That disagreement is exactly what wakes the VLM up.
5. The VLM looked at the cropped image, and said no.

**THE FINDING, in one sentence:** the VLM is not a general-purpose "recovery" system — it's a **safety check on the fast detector's mistakes**, and it only earns its 333 milliseconds when there's something in the scene that can fool the detector.

This is a **better** result than "the VLM improves recovery," for three reasons:
- It has a **mechanism**. I can say exactly why it helped, not just that it did.
- It explains **all 48 trials**, not just the 6 that differed. The 42 identical trials stop being a disappointing null result and become part of the explanation: on a bare desk there was **no mistake available for the VLM to catch**.
- It's **actionable**. It tells an engineer exactly when to pay for a VLM and when to skip it.

---
# PART 6 — Things I am NOT allowed to claim
This section exists because it's the difference between a project that survives questioning and one that doesn't. If I don't write these down, my mentor will find them — and it's much better if I found them first.

| Tempting claim | Why I can't say it |
| :- | :- |
| "The VLM cut false belief by 34%" | 42 of 48 trials were identical, and two trials carry 62% of the difference. That's not an average effect, it's a handful of episodes. I can report the direction (6 out of 6, p = 0.016) and nothing more. |
| "The system recovers from occlusion" | This one stings. **occlude_hand** and **occlude_object** — the two scenarios my research question is literally named after — show **zero** false belief with OR without the VLM. The VLM contributes nothing there. The reason the system survives a hand covering the cup is a simple counter: it holds its belief for 12 frames before giving up. That's a timer, not reasoning. |
| "The VLM verifies identity" | On the identity_swap trials, turning the VLM on cut lying from 76 frames to 36 — but accuracy **went down**, from 0.30 to 0.18. That means it wasn't correctly identifying the cup. It was *refusing to commit* to almost anything, and since the truth was usually "absent," refusing happened to be right more often. That's suppression, not verification. Reporting the 76→36 without this would be misleading. |
| "Clutter is a clean independent variable" | Clutter and detector-reliability are tangled together on purpose — clutter breaks the detector, and that's literally how it has its effect. That's fine, but I have to say it out loud rather than pretend clutter is a separate, independent thing. |
| "This holds generally" | 48 trials means only 6 per scenario and 3 per scenario-and-clutter combination. Three data points is enough to notice a pattern, not enough to prove one. The per-scenario tables describe what happened; they don't predict what would happen next time. |

---
# PART 7 — The other things worth keeping
### 1. The speed design works, and this is the strongest fully-supported result
The VLM ran on **1.6% of frames** (135 calls out of 8,428). Frames where it *didn't* run cost 29 ms. Frames where it did cost 334 ms. So the system runs fast almost always, and only pays the expensive price when something has actually gone wrong. If I'd called the VLM every frame the robot would run about 10× slower for no measurable benefit on a clean desk.

### 2. The "hold on for 12 frames" timer is a real dial with a real trade-off
When the cup vanishes, the system keeps believing for a while before giving up. I re-ran the data with that hold-time set to 12, 30 and 60 frames:

| Hold for... | Frames it LIED | Frames it lost track | Accuracy |
| :- | :- | :- | :- |
| 12 frames (what I used) | **196** | 711 | 0.640 |
| 30 frames | 229 | 478 | 0.762 |
| 60 frames | 235 | 232 | 0.890 |

Holding on longer makes it look much more accurate and much less "blind" — but it lies more. There's no universally correct setting. If the robot is about to *grab* something, lying is far worse than hesitating, so a short hold is right. If it's just watching, a long hold looks better. **Being able to show this trade-off is more useful than picking a winner.**

### 3. Environment changes are completely solved
All 12 lighting and camera-movement trials were perfect, with and without the VLM. Turning the lights off didn't fool it, and when the arm moved itself the system didn't panic — because the code tells the tracker "that was me" before it moves. Worth noting that this is solved by ordinary engineering, not by AI.

---
# PART 8 — What to do next
### Priority 1 — run more trials on the three scenarios that matter (about 35 minutes)
`python final_experiment.py --reps 8 --clutter none,high --scenarios removal,substitution,identity_swap`

Everything interesting happened in **removal**, **substitution** and **identity_swap**. The other five scenarios are either perfect every time or never produce a difference. Running 8 repeats of just those three triples the data exactly where it's needed and turns "6 out of 6, direction only" into a number with an actual size.

### Priority 2 — save a picture of the detector making the mistake
Right now I can prove the detector was wrong 62.6% of the time with distractors, but I can't *show* it, because I only logged whether it fired and how confident it was — not which object it boxed. One screenshot of a green box drawn around a blue mug, labelled "red cup, confidence 0.58", would be the most convincing single image this project could produce. It's a small code change.

### Priority 3 — the question my mentor will ask, so I should answer it first
"Could you just make the detector less trigger-happy instead?" The detector reports a confidence score, and I could simply ignore anything below, say, 0.5. If that alone removes most of the false detections, then a one-line change buys most of what the VLM buys, for **zero** extra time — and the case for the VLM has to rest on the cases a threshold *can't* fix. I'd much rather find this out myself than be asked it in a meeting with no answer.

---
# PART 9 — Where the data lives
- Everything is in runs/2026-09-01_233533/. Every frame was saved as an image, raw and annotated. Nothing was ever overwritten.
- results.csv — the 48-row summary table. results.json — 4.5 MB, the robot's belief at every single frame. run_log.txt — a timestamped log of the session. ablation_scores.json — the with/without-VLM comparison.
- final_experiment.py.snapshot — a frozen copy of the *exact* code that produced this data, saved automatically at the start of the run. So even if I change the code later, anyone can see precisely what was run.
- Before the trials started, a baseline check confirmed the cup was detected in 10 out of 10 frames and the image brightness was normal.
- Two trials (3 and 6) were re-staged because I fumbled the setup. Both re-stagings are recorded in the log, and the discarded attempts were **not** scored.
- Hardware: Arduino Uno + PCA9685 servo driver, 2-motor arm returned to its home position before every trial; Logitech C270 webcam at 480×360; the VLM running on a 6 GB graphics card.
- Code: github.com/Alm1973/avi-occlusion-recovery

---
# Glossary — every technical word in one place
| Word | Plain meaning |
| :- | :- |
| **Ablation** | Removing one part of a system and re-running everything, to find out what that part was contributing. |
| **Accuracy** | Fraction of graded frames where the robot's belief matched reality. 1.0 = perfect. |
| **Confidence score** | A number from 0 to 1 that a detector attaches to each guess, meaning "how sure am I". 0.9 = very sure, 0.2 = barely. |
| **Counterbalancing** | Shuffling the order of tests so that improvement over time can't be mistaken for a real effect. |
| **Edge frame** | A frame captured right at the moment the scene is changing, when even a human couldn't say what's true. Recorded, but excluded from scoring. |
| **Event-gating** | Only running the expensive part of the system when something has actually gone wrong, instead of all the time. |
| **False belief** | The robot claiming the target is there when it isn't. The dangerous mistake — the main thing this experiment measures. |
| **False positive** | Any detection of something that isn't actually there. |
| **Frame** | One single photo from the video. This system runs at about 28 frames per second. |
| **Ground truth** | What was *actually* true, recorded independently of what the robot thought. |
| **Millisecond (ms)** | One thousandth of a second. 1000 ms = 1 second. |
| **n** | The number of data points. "n = 48" means 48 trials. |
| **Occlusion** | Something blocking the view of the target. A hand in front of the cup occludes it. |
| **OpenCV** | A standard programming library for image maths — measuring positions, sizes, overlaps. Not AI. |
| **p-value** | The probability of getting a result this lopsided purely by luck, if the thing being tested actually did nothing. Below 0.05 is the usual bar. |
| **Pre-registration** | Writing down what counts as success *before* looking at any results, so you can't move the goalposts. |
| **Sign test** | A simple statistical test that ignores the size of each change and only counts which direction it went. Useful when a few extreme values would otherwise dominate. |
| **Spurious reacquisition** | The robot wrongly announcing it has "found the cup again" when the cup is actually gone. |
| **VLM (Vision-Language Model)** | An AI that can look at an image and answer questions about it in plain English. Like ChatGPT, but it can see. |
| **YOLO-World** | The specific fast detector used here. You give it a phrase like "red cup" and it draws boxes around matching things, many times per second. |
