import re

import cv2
import numpy as np

CUE_NAMES = ["colour", "shape", "reflectivity", "size", "brightness"]

CUE_KEYWORDS = {
    "reflectivity": ["shiny", "shiniest", "reflective", "metallic", "glossy",
                     "polished", "mirror", "chrome", "glinting", "sparkly"],
    "shape": ["round", "circular", "square", "rectangular", "tall", "flat",
              "long", "thin", "wide", "ball", "sphere", "box", "cube", "disc"],
    "size": ["big", "biggest", "large", "largest", "small", "smallest",
             "tiny", "huge"],
    "brightness": ["bright", "brightest", "dark", "darkest", "glowing", "lit"],
    "colour": ["red", "green", "blue", "yellow", "orange", "purple", "pink",
               "teal", "cyan", "white", "black", "grey", "gray", "brown",
               "silver", "colour", "color"],
}

SUPERLATIVES = ["shiniest", "biggest", "largest", "smallest", "brightest",
                "darkest", "roundest", "reddest", "bluest", "greenest",
                "most reflective", "most colourful", "most colorful"]

CUE_PROMPT = (
    "A robot must find: \"{q}\".\n"
    "Which visual properties should it use to find this? Choose from: "
    "colour, shape, reflectivity, size, brightness.\n"
    "Reply with ONLY a comma-separated list of the properties that matter "
    "most for this specific request, best first."
)


def parse_cues(text, query):
    picked = []
    t = (text or "").lower()
    for name in CUE_NAMES:
        alt = "color" if name == "colour" else name
        if re.search(rf"\b{name}\b", t) or re.search(rf"\b{alt}\b", t):
            picked.append(name)

    q = query.lower()
    for cue, words in CUE_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            if cue not in picked:
                picked.append(cue)

    if "shape" not in picked and shape_target(query):
        picked.append("shape")
    return picked or ["colour", "shape"]


def is_superlative(query):
    q = query.lower()
    return any(s in q for s in SUPERLATIVES) or bool(
        re.search(r"\b(most|least)\b", q))


def shape_features(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.contourArea(cv2.convexHull(contour))
    return {
        "circularity": (4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0,
        "aspect": (w / h) if h else 0.0,
        "solidity": (area / hull) if hull else 0.0,
        "extent": (area / (w * h)) if w * h else 0.0,
        "area": area,
    }


def reflectivity(frame_bgr, contour):
    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    n = int(mask.sum() // 255)
    if n < 20:
        return {"specular_frac": 0.0, "max_v": 0.0, "v_std": 0.0}
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][mask > 0].astype(np.float32)
    s = hsv[:, :, 1][mask > 0].astype(np.float32)
    spec = float(((v > 230) & (s < 60)).mean())
    return {"specular_frac": spec, "max_v": float(v.max()),
            "v_std": float(v.std())}


def brightness(frame_bgr, contour):
    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    if mask.sum() == 0:
        return {"mean_v": 0.0}
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return {"mean_v": float(hsv[:, :, 2][mask > 0].mean())}


def measure_all(frame_bgr, contour, cue_list):
    feats = {}
    if "shape" in cue_list:
        feats.update(shape_features(contour))
    else:
        feats["area"] = cv2.contourArea(contour)
    if "reflectivity" in cue_list:
        feats.update(reflectivity(frame_bgr, contour))
    if "brightness" in cue_list:
        feats.update(brightness(frame_bgr, contour))
    return feats


def shape_target(query):
    q = query.lower()
    if any(w in q for w in ("ball", "sphere", "round", "circular", "disc",
                            "circle")):
        return {"circularity": 0.85, "aspect": 1.0}
    if any(w in q for w in ("box", "cube", "square", "rectangular", "book")):
        return {"circularity": 0.55, "aspect": 1.0, "extent": 0.85}
    if any(w in q for w in ("bottle", "can", "cup", "mug", "glass", "tall")):
        return {"aspect": 0.5}
    if any(w in q for w in ("flat", "wide", "keyboard", "laptop", "paper")):
        return {"aspect": 2.5}
    return {}


def score_candidate(feats, query, cue_list):
    scores = []
    if "shape" in cue_list:
        tgt = shape_target(query)
        if "circularity" in tgt and "circularity" in feats:
            scores.append(max(0.0, 1.0 - abs(feats["circularity"]
                                             - tgt["circularity"])))
        if "aspect" in tgt and feats.get("aspect"):
            a, t = feats["aspect"], tgt["aspect"]
            scores.append(max(0.0, 1.0 - abs(np.log((a + 1e-6) / t)) / 1.6))
        if "extent" in tgt and "extent" in feats:
            scores.append(max(0.0, 1.0 - abs(feats["extent"] - tgt["extent"])))
    if "reflectivity" in cue_list:
        scores.append(min(1.0, feats.get("specular_frac", 0.0) / 0.15))
    if "brightness" in cue_list:
        dark = "dark" in query.lower()
        mv = feats.get("mean_v", 0.0) / 255.0
        scores.append(1.0 - mv if dark else mv)
    if "size" in cue_list:
        small = any(w in query.lower() for w in ("small", "smallest", "tiny"))
        scores.append(0.5)
    return float(np.mean(scores)) if scores else 0.5


def rank_candidates(frame_bgr, candidates, query, cue_list):
    out = []
    for c in candidates:
        feats = measure_all(frame_bgr, c["contour"], cue_list)
        item = dict(c)
        item["features"] = feats
        item["score"] = score_candidate(feats, query, cue_list)
        out.append(item)

    if "size" in cue_list and out:
        areas = [i["features"].get("area", 0) for i in out]
        amax = max(areas) or 1
        small = any(w in query.lower() for w in ("small", "smallest", "tiny"))
        for i in out:
            rel = i["features"].get("area", 0) / amax
            i["score"] = (i["score"] + (1 - rel if small else rel)) / 2

    out.sort(key=lambda i: i["score"], reverse=True)
    return out
