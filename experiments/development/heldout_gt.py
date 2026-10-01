
HELDOUT_GT = {
    "c2_05_pan_b108_t90.jpg": {
        "laptop": 1, "keyboard": 1, "computer mouse": 1, "water bottle": 1,
    },
    "c2_07_tilt_b90_t78.jpg": {
        "laptop": 1, "keyboard": 0, "computer mouse": 0, "water bottle": 1,
    },
    "c2_09_tilt_b90_t108.jpg": {
        "laptop": 0, "keyboard": 1, "computer mouse": 0, "water bottle": 0,
    },
    "c2_10_home.jpg": {
        "laptop": 1, "keyboard": 1, "computer mouse": 1, "water bottle": 1,
    },
}

HALLUCINATION_PROBES = {
    "banana": ["banana"],
    "bicycle": ["bicycle", "bike"],
    "elephant": ["elephant"],
    "umbrella": ["umbrella"],
}


def class_balance():
    out = {}
    for labels in HELDOUT_GT.values():
        for obj, v in labels.items():
            p, n = out.get(obj, (0, 0))
            out[obj] = (p + v, n + (1 - v))
    return out


if __name__ == "__main__":
    print(f"{len(HELDOUT_GT)} frames, "
          f"{sum(len(v) for v in HELDOUT_GT.values())} labelled points")
    for obj, (p, n) in class_balance().items():
        print(f"  {obj:<15} present {p}  absent {n}")
