"""
moondream_spatial_probe.py — Assignment 3(b)

Probe Moondream's raw spatial reasoning ability, independent of the
OpenCV pipeline, so you know how much you can safely offload to it.
See docs/vlm-spatial-reasoning-review.md for the protocol and results table.

Usage:
    1. Capture 3-5 still frames with a known object at a known position
       and save them under code/hardware_tests/frames/.
    2. Fill in the calls at the bottom with your actual filenames and
       ground-truth positions.
    3. python moondream_spatial_probe.py
    4. Copy the printed results into the table in
       docs/vlm-spatial-reasoning-review.md.
"""

import ollama

QUESTIONS = [
    "Where in the frame is the {color} object - left, center, or right?",
    "Is the {color} object closer to the top or bottom of the image?",
]

TWO_OBJECT_QUESTION = (
    "There are two objects in the image, a {color_a} one and a {color_b} one. "
    "Which one is further to the left?"
)


def probe_single(image_path, color, true_lr, true_tb):
    results = {}
    for template in QUESTIONS:
        q = template.format(color=color)
        response = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": q, "images": [image_path]}],
        )
        results[q] = response["message"]["content"].strip()

    print(f"--- {image_path} ---")
    for q, a in results.items():
        print(f"Q: {q}\nA: {a}")
    print(f"(Expected: left/right={true_lr}, top/bottom={true_tb})\n")
    return results


def probe_two_object(image_path, color_a, color_b, true_answer):
    q = TWO_OBJECT_QUESTION.format(color_a=color_a, color_b=color_b)
    response = ollama.chat(
        model="moondream",
        messages=[{"role": "user", "content": q, "images": [image_path]}],
    )
    answer = response["message"]["content"].strip()
    print(f"--- {image_path} ---\nQ: {q}\nA: {answer}\n(Expected: {true_answer})\n")
    return answer


if __name__ == "__main__":
    # TODO(Shaurya): replace with your actual captured frames + ground truth
    probe_single("frames/red_left.jpg", "red", "left", "n/a")
    probe_single("frames/red_center.jpg", "red", "center", "n/a")
    probe_single("frames/red_right.jpg", "red", "right", "n/a")
    # probe_two_object("frames/red_and_blue.jpg", "red", "blue", "red")
