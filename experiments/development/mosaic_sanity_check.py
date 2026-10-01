
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

RESIZED_OUTPUT_PATH = "mosaic_resized_sanity_check.jpg"
TARGET_SIZE = (480, 360)

SANITY_CHECK_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 200,
}

DESCRIBE_PROMPT = (
    "This image is a grid mosaic made of small photo tiles, with plain "
    "gray squares filling in the unused parts of the grid. Describe what "
    "you can see: how many distinct photo tiles (non-gray) can you make "
    "out, roughly where are they in the grid, and can you identify any "
    "objects in them (e.g. a red cup, a hand, a keyboard, a person)? Be "
    "specific about what is and isn't visible."
)


def main():
    tile_map = TileMap()

    for path, base_angle, tilt_angle in TEST_FRAMES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping {path} (failed to load)")
            continue
        bucket = tile_map.update(frame, base_angle, tilt_angle)
        print(f"{path} @ base={base_angle} tilt={tilt_angle} -> bucket {bucket}")

    mosaic = tile_map.build_mosaic()
    print(f"Mosaic built: {mosaic.shape[1]}x{mosaic.shape[0]}, "
          f"coverage {tile_map.coverage() * 100:.1f}%")

    resized = cv2.resize(mosaic, TARGET_SIZE)
    cv2.imwrite(RESIZED_OUTPUT_PATH, resized)
    print(f"Saved resized ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}) mosaic to "
          f"{RESIZED_OUTPUT_PATH}")

    success, buffer = cv2.imencode(".jpg", resized)
    if not success:
        print("Failed to encode resized mosaic")
        return

    print("Asking model to describe the resized mosaic...")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": DESCRIBE_PROMPT, "images": [buffer.tobytes()]}],
        keep_alive="30m",
        options=SANITY_CHECK_OPTIONS,
    )
    description = response["message"]["content"].strip()
    print("\n=== MODEL DESCRIPTION ===")
    print(description)


if __name__ == "__main__":
    main()
