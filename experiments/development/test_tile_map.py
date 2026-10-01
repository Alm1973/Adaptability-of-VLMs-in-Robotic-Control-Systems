
import cv2

from tile_map import TileMap

TEST_FRAMES = [
    ("index1_test2.jpg", 90, 90),
    ("hand_test.jpg", 30, 60),
    ("index0_test.jpg", 150, 120),
    ("index1_test2.jpg", 15, 165),
    ("index1_test2.jpg", 165, 15),
]

OUTPUT_PATH = "mosaic_preview.jpg"


def main():
    tile_map = TileMap()

    for path, base_angle, tilt_angle in TEST_FRAMES:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping {path} (failed to load)")
            continue

        bucket = tile_map.update(frame, base_angle, tilt_angle)
        print(f"{path} @ base={base_angle} tilt={tilt_angle} -> bucket {bucket}")

    print(f"Coverage: {tile_map.coverage() * 100:.1f}% "
          f"({len(tile_map.tiles)}/{tile_map.num_buckets ** 2} buckets)")

    mosaic = tile_map.build_mosaic()
    cv2.imwrite(OUTPUT_PATH, mosaic)
    print(f"Saved {OUTPUT_PATH} ({mosaic.shape[1]}x{mosaic.shape[0]})")


if __name__ == "__main__":
    main()
