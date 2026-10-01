
import cv2
import ollama

from tile_map import TileMap
from vlm_recovery import MODEL_NAME

TEST_FRAMES = [
    ("index1_test2.jpg", 15, 15),
    ("index1_test2.jpg", 165, 165),
    ("index1_test2.jpg", 90, 90),
    ("index1_test2.jpg", 75, 45),
    ("hand_test.jpg", 45, 135),
    ("hand_test.jpg", 135, 45),
    ("hand_test.jpg", 105, 135),
    ("index0_test.jpg", 15, 165),
    ("index0_test.jpg", 165, 15),
]

TARGET_SIZE = (480, 360)
NUM_RUNS = 3

DIRECTION_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 10,
}

DIRECTION_PROMPT = (
    "This is a map of what's been explored so far -- gray squares are "
    "unexplored, real photos are explored. Looking at where the gray "
    "(unexplored) squares are relative to the photographed squares, which "
    "general direction has the most unexplored territory: left, right, up, "
    "or down? Respond with exactly one word: left, right, up, or down."
)

VALID_WORDS = ["left", "right", "up", "down"]


def extract_direction(raw_text):
    text = raw_text.lower()
    found = [w for w in VALID_WORDS if w in text]
    if not found:
        return None
    found.sort(key=lambda w: text.rfind(w))
    return found[-1]


def main():
    tile_map = TileMap()

    for path, base_angle, tilt_angle in TEST_FRAMES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping {path} (failed to load)")
            continue
        tile_map.update(frame, base_angle, tilt_angle)

    mosaic = tile_map.build_mosaic()
    resized = cv2.resize(mosaic, TARGET_SIZE)
    print(f"Mosaic: {mosaic.shape[1]}x{mosaic.shape[0]} -> resized to "
          f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}, coverage {tile_map.coverage()*100:.1f}%")

    success, buffer = cv2.imencode(".jpg", resized)
    if not success:
        print("Failed to encode resized mosaic")
        return
    image_bytes = buffer.tobytes()

    print(f"\nRunning the direction probe {NUM_RUNS}x on the identical image...\n")

    results = []
    for i in range(1, NUM_RUNS + 1):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": DIRECTION_PROMPT, "images": [image_bytes]}],
            keep_alive="30m",
            options=DIRECTION_OPTIONS,
        )
        raw_text = response["message"]["content"].strip()
        parsed = extract_direction(raw_text)
        results.append(parsed)
        print(f"Run {i}/{NUM_RUNS}: raw={raw_text!r} -> parsed={parsed}")

    print(f"\n=== RESULTS: {results} ===")
    unique = set(results)
    if len(unique) == 1 and None not in unique:
        print(f"Consistent: all {NUM_RUNS} runs agreed on '{results[0]}'")
    else:
        print(f"INCONSISTENT: {len(unique)} distinct answers across {NUM_RUNS} runs")


if __name__ == "__main__":
    main()
