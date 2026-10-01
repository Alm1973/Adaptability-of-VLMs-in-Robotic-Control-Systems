import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import llm_backend
from cues import CUE_PROMPT, is_superlative, parse_cues
from cues import rank_candidates as cue_rank
from open_vocab_detect import LIST_PROMPT as _OV_LIST_PROMPT
from open_vocab_detect import _dedupe as _ov_dedupe
from open_vocab_detect import _parse_items as _ov_parse
from open_vocab_detect import is_present as _ov_present

GRID_CELLS = [
    "top-left", "top-center", "top-right",
    "middle-left", "middle-center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]

HUE_PRIOR = {
    "red": 0, "orange": 12, "yellow": 27, "green": 60, "lime": 45,
    "teal": 90, "cyan": 92, "turquoise": 90, "blue": 112, "navy": 118,
    "purple": 140, "violet": 140, "magenta": 155, "pink": 168, "brown": 12,
}
ACHROMATIC = {"white", "black", "gray", "grey", "silver", "clear",
              "transparent"}

LOCATE_PROMPT = (
    "Look at this image as a 3x3 grid. Which cell contains the {obj}? "
    "Answer with EXACTLY one of: top-left, top-center, top-right, "
    "middle-left, middle-center, middle-right, bottom-left, bottom-center, "
    "bottom-right. If the {obj} is not visible at all, answer: none."
)
COLOR_PROMPT = (
    "What is the main colour of the {obj} in this image? Answer with ONLY "
    "one colour word (for example: red, blue, teal, white, black)."
)
VERIFY_PROMPT = (
    "Is this a {obj}? Look carefully at the object in the centre. "
    "Answer with only one word: yes or no."
)

CROP_PAD_RATIO = 0.5


@dataclass
class DetectorConfig:
    object_name: str
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    min_area: int
    max_area: int
    aspect_lo: float
    aspect_hi: float
    solidity_lo: float
    cell: str = ""
    color_name: str = ""
    achromatic: bool = False
    sample_px: int = 0
    cues: list = field(default_factory=lambda: ["colour"])
    superlative: bool = False
    hsv_ranges: list = None
    notes: list = field(default_factory=list)

    def describe(self):
        return (f"{self.object_name}: H[{self.hsv_lower[0]}-{self.hsv_upper[0]}] "
                f"S[{self.hsv_lower[1]}-{self.hsv_upper[1]}] "
                f"V[{self.hsv_lower[2]}-{self.hsv_upper[2]}] "
                f"area[{self.min_area}-{self.max_area}] "
                f"aspect[{self.aspect_lo:.2f}-{self.aspect_hi:.2f}] "
                f"solidity>={self.solidity_lo:.2f} "
                f"(colour '{self.color_name}', cell {self.cell}, "
                f"{self.sample_px}px sampled)")


def _ask(prompt, image, max_tokens=16):
    out = llm_backend.vision_chat(prompt, image, max_tokens=max_tokens)
    return (out or "").strip().lower()


def _parse_cell(text):
    if "none" in text and not any(c in text for c in GRID_CELLS):
        return None
    for cell in sorted(GRID_CELLS, key=len, reverse=True):
        if cell in text:
            return cell
    t = text.replace(" ", "-")
    for cell in GRID_CELLS:
        if cell in t:
            return cell
    return None


def _cell_roi(frame, cell):
    h, w = frame.shape[:2]
    row = 0 if cell.startswith("top") else (1 if cell.startswith("middle") else 2)
    col = 0 if cell.endswith("left") else (1 if cell.endswith("center") else 2)
    y0, y1 = int(h * row / 3), int(h * (row + 1) / 3)
    x0, x1 = int(w * col / 3), int(w * (col + 1) / 3)
    return (x0, y0, x1, y1)


def _dominant_clusters(hsv_roi, k=4):
    pts = hsv_roi.reshape(-1, 3).astype(np.float32)
    if len(pts) > 20000:
        idx = np.random.RandomState(0).choice(len(pts), 20000, replace=False)
        sample = pts[idx]
    else:
        sample = pts
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(k, max(1, len(sample) // 50))
    _, labels, centers = cv2.kmeans(sample, k, None, crit, 3,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    return centers, counts, sample, labels.flatten()


def _hue_distance(h1, h2):
    d = abs(float(h1) - float(h2))
    return min(d, 180 - d)


def _rank_clusters(centers, counts, color_name):
    scored = []
    for i, c in enumerate(centers):
        if color_name in ACHROMATIC:
            s_pen = float(c[1])
            v_pref = -float(c[2]) if color_name in (
                "white", "silver", "clear", "transparent") else float(c[2])
            score = s_pen + v_pref * 0.5 - counts[i] * 0.001
        else:
            prior = HUE_PRIOR.get(color_name)
            if prior is None:
                score = -counts[i]
            else:
                sat_pen = max(0.0, 40.0 - float(c[1])) * 2.0
                dark_pen = max(0.0, 60.0 - float(c[2])) * 1.5
                score = (_hue_distance(c[0], prior) + sat_pen + dark_pen
                         - counts[i] * 0.0005)
        scored.append((score, i))
    scored.sort()
    return [i for _, i in scored]


def _pick_cluster(centers, counts, color_name):
    if color_name in ACHROMATIC:
        scores = []
        for i, c in enumerate(centers):
            s_pen = c[1]
            v_pref = -c[2] if color_name in ("white", "silver", "clear",
                                             "transparent") else c[2]
            scores.append(s_pen + v_pref * 0.5 - counts[i] * 0.001)
        return int(np.argmin(scores))
    prior = HUE_PRIOR.get(color_name)
    if prior is None:
        return int(np.argmax(counts))
    best, best_score = 0, 1e9
    for i, c in enumerate(centers):
        if c[1] < 40:
            continue
        score = _hue_distance(c[0], prior) - counts[i] * 0.0005
        if score < best_score:
            best, best_score = i, score
    return int(best)


def _measure_ranges(pixels, achromatic):
    h = pixels[:, 0].astype(np.float32)
    s = pixels[:, 1]
    v = pixels[:, 2]
    if achromatic:
        lo = np.array([0, 0, max(0, np.percentile(v, 5) - 25)])
        hi = np.array([179, min(255, np.percentile(s, 95) + 40), 255])
        return [(lo.astype(np.uint8), hi.astype(np.uint8))]

    s_lo = max(30, np.percentile(s, 10) - 40)
    v_lo = max(30, np.percentile(v, 10) - 40)
    s_hi = min(255, np.percentile(s, 90) + 50)
    v_hi = min(255, np.percentile(v, 90) + 50)

    ang = h * 2.0 * np.pi / 180.0
    mean_ang = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    mean_h = (mean_ang * 180.0 / (2.0 * np.pi)) % 180.0
    shift = 90.0 - mean_h
    h_rot = (h + shift) % 180.0
    pad = 8.0
    lo_r = np.percentile(h_rot, 5) - pad
    hi_r = np.percentile(h_rot, 95) + pad
    if hi_r - lo_r > 50:
        lo_r, hi_r = 90.0 - 25.0, 90.0 + 25.0
    lo_h = (lo_r - shift) % 180.0
    hi_h = (hi_r - shift) % 180.0

    def rng(a, b):
        return (np.array([a, s_lo, v_lo], np.uint8),
                np.array([b, s_hi, v_hi], np.uint8))

    if lo_h <= hi_h:
        return [rng(lo_h, hi_h)]
    return [rng(0, hi_h), rng(lo_h, 179)]


def configure(frame, object_name, verbose=True, cluster_rank=0):
    t0 = time.time()

    cue_txt = _ask(CUE_PROMPT.format(q=object_name), frame, max_tokens=24)
    cue_list = parse_cues(cue_txt, object_name)
    superlative = is_superlative(object_name)

    loc = _ask(LOCATE_PROMPT.format(obj=object_name), frame)
    cell = _parse_cell(loc)
    if cell is None:
        if verbose:
            print(f"[CONFIG] VLM could not locate '{object_name}' "
                  f"(said {loc!r})")
        return None, time.time() - t0

    x0, y0, x1, y1 = _cell_roi(frame, cell)
    roi = frame[y0:y1, x0:x1]
    named = [w for w in re.findall(r"[a-z]+", object_name.lower())
             if w in HUE_PRIOR or w in ACHROMATIC]
    if named:
        color = named[0]
    else:
        color = _ask(COLOR_PROMPT.format(obj=object_name), frame)
        color = re.sub(r"[^a-z]", "", color.split()[0]) if color.split() else ""

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    centers, counts, sample, labels = _dominant_clusters(hsv_roi)
    order = _rank_clusters(centers, counts, color)
    idx = order[min(cluster_rank, len(order) - 1)]
    pixels = sample[labels == idx]
    if len(pixels) < 30:
        pixels = sample
    achro = color in ACHROMATIC

    if not achro and color in HUE_PRIOR and len(pixels):
        med_h = float(np.median(pixels[:, 0]))
        if _hue_distance(med_h, HUE_PRIOR[color]) > 25:
            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hch = hsv_full[:, :, 0].astype(np.float32)
            d = np.abs(hch - HUE_PRIOR[color])
            d = np.minimum(d, 180 - d)
            m = ((d <= 15) & (hsv_full[:, :, 1] >= 70)
                 & (hsv_full[:, :, 2] >= 60))
            if int(m.sum()) >= 200:
                px = hsv_full[m].astype(np.float32)
                if len(px) > 20000:
                    px = px[np.random.RandomState(0).choice(
                        len(px), 20000, replace=False)]
                pixels = px
                if verbose:
                    print(f"[CONFIG] cell pixels were not '{color}' "
                          f"(H~{med_h:.0f}) -> re-measured from "
                          f"{int(m.sum())} whole-frame '{color}' pixels")

    ranges = _measure_ranges(pixels, achro)
    lo, hi = ranges[0]

    if achro and "colour" in cue_list:
        cue_list = [c for c in cue_list if c != "colour"]
        if "reflectivity" not in cue_list:
            cue_list.insert(0, "reflectivity")
        if "shape" not in cue_list:
            cue_list.append("shape")
        if verbose:
            print(f"[CONFIG] '{color}' is achromatic -> dropping colour cue, "
                  f"using {cue_list} with edge proposals")

    mask = None
    for rlo, rhi in ranges:
        cur = cv2.inRange(hsv_roi, rlo, rhi)
        mask = cur if mask is None else cv2.bitwise_or(mask, cur)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(cv2.erode(mask, kernel, 1), kernel, 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = aspect = solidity = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 1.0
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull if hull else 1.0

    frame_area = frame.shape[0] * frame.shape[1]
    if area and area > 50:
        min_area = int(max(150, area * 0.15))
        max_area = int(min(frame_area * 0.9, area * 12))
        aspect_lo, aspect_hi = max(0.15, aspect * 0.35), min(8.0, aspect * 2.8)
        sol_lo = max(0.20, (solidity or 0.6) * 0.55)
    else:
        min_area, max_area = 300, int(frame_area * 0.5)
        aspect_lo, aspect_hi, sol_lo = 0.2, 6.0, 0.3

    cfg = DetectorConfig(
        object_name=object_name, hsv_lower=lo, hsv_upper=hi,
        min_area=min_area, max_area=max_area,
        aspect_lo=aspect_lo, aspect_hi=aspect_hi, solidity_lo=sol_lo,
        cell=cell, color_name=color, achromatic=achro, sample_px=len(pixels),
        cues=cue_list, superlative=superlative,
    )
    cfg.hsv_ranges = ranges
    if len(ranges) > 1:
        cfg.notes.append("hue_wraps:2_ranges")

    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_mask = None
    for rlo, rhi in ranges:
        cur = cv2.inRange(hsv_full, rlo, rhi)
        full_mask = cur if full_mask is None else cv2.bitwise_or(full_mask, cur)
    coverage = float((full_mask > 0).mean())
    cfg.notes.append(f"frame_coverage={coverage:.3f}")
    if coverage > 0.35 and "colour" in cue_list:
        cfg.notes.append("TOO_PERMISSIVE")
        if verbose:
            print(f"[CONFIG] WARNING: range covers {coverage*100:.0f}% of the "
                  f"frame -- too permissive to track on "
                  f"(colour '{color}' is likely background, not the object)")
    dt = time.time() - t0
    if verbose:
        print(f"[CONFIG] {cfg.describe()}  ({dt:.2f}s, 2 VLM calls)")
    return cfg, dt


def configure_validated(frame, object_name, max_attempts=3, verbose=True,
                        allow_fallback=True):
    total = 0.0
    best_unverified = None
    for attempt in range(max_attempts):
        cfg, dt = configure(frame, object_name, verbose=verbose,
                            cluster_rank=attempt)
        total += dt
        if cfg is None:
            return None, total
        if "TOO_PERMISSIVE" in cfg.notes:
            if verbose:
                print(f"[CONFIG] attempt {attempt+1}: too permissive, re-tuning")
            best_unverified = best_unverified or cfg
            continue
        cands = detect_candidates(frame, cfg)
        if not cands:
            if verbose:
                print(f"[CONFIG] attempt {attempt+1}: range finds nothing, "
                      f"re-tuning")
            best_unverified = best_unverified or cfg
            continue
        ok, vdt = verify(frame, cands[0]["box"], object_name)
        total += vdt
        if ok:
            cfg.notes.append(f"validated_attempt={attempt+1}")
            if verbose:
                print(f"[CONFIG] ✅ validated on attempt {attempt+1} "
                      f"({total:.2f}s total)")
            return cfg, total
        if verbose:
            print(f"[CONFIG] attempt {attempt+1}: verifier rejected the top "
                  f"candidate -> re-tuning on next cluster")
        best_unverified = best_unverified or cfg
    if not allow_fallback:
        if verbose:
            print(f"[CONFIG] no cluster validated; refusing fallback")
        return None, total
    if verbose:
        print(f"[CONFIG] no cluster validated; falling back to first config")
    return best_unverified, total


def propose_generic(frame, min_area=400):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= min_area]


def detect_candidates(frame, cfg, suppression=None, max_candidates=5):
    cue_list = getattr(cfg, "cues", None) or ["colour"]
    use_colour = "colour" in cue_list and cfg.hsv_lower is not None

    if use_colour:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = cfg.hsv_ranges or [(cfg.hsv_lower, cfg.hsv_upper)]
        mask = None
        for rlo, rhi in ranges:
            cur = cv2.inRange(hsv, rlo, rhi)
            mask = cur if mask is None else cv2.bitwise_or(mask, cur)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    else:
        cnts = propose_generic(frame, min_area=max(200, cfg.min_area // 2))

    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < cfg.min_area:
            break
        if area > cfg.max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 0
        if not (cfg.aspect_lo <= aspect <= cfg.aspect_hi):
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull if hull else 0
        if sol < cfg.solidity_lo:
            continue
        cx, cy = x + w // 2, y + h // 2
        if suppression and suppression.is_suppressed(cx, cy):
            continue
        out.append({"box": (x, y, w, h), "center": (cx, cy),
                    "area": area, "aspect": aspect, "solidity": sol,
                    "contour": c})
        if len(out) >= max_candidates * 3:
            break

    if out and cue_list != ["colour"]:
        out = cue_rank(frame, out, cfg.object_name, cue_list)
    return out[:max_candidates]


def _crop_with_padding(frame, box, ratio=CROP_PAD_RATIO):
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    px, py = int(w * ratio), int(h * ratio)
    return frame[max(0, y - py):min(fh, y + h + py),
                 max(0, x - px):min(fw, x + w + px)]


VERIFY_LIST_SAMPLES = 2


def verify(frame, box, object_name):
    crop = _crop_with_padding(frame, box)
    t0 = time.time()
    if crop.size == 0:
        return None, 0.0
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 64:
        scale = max(2.0, 128.0 / max(1, min(ch, cw)))
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    words = re.findall(r"[a-z]+", object_name.lower())
    queries = [object_name] + ([words[-1]] if len(words) > 1 else [])
    for _ in range(VERIFY_LIST_SAMPLES):
        out = llm_backend.vision_chat(_OV_LIST_PROMPT, crop, max_tokens=60)
        items = _ov_dedupe(_ov_parse(out)) if out else []
        if any(_ov_present(q, items) for q in queries):
            return True, time.time() - t0
    return False, time.time() - t0


class Suppression:

    def __init__(self, radius_px=100, expiry_sec=15):
        self.radius = radius_px
        self.expiry = expiry_sec
        self.zones = []

    def add(self, center):
        self.zones.append((center[0], center[1], time.time()))

    def is_suppressed(self, cx, cy):
        now = time.time()
        self.zones = [z for z in self.zones if now - z[2] < self.expiry]
        return any(((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5 <= self.radius
                   for zx, zy, _ in self.zones)


if __name__ == "__main__":
    import glob
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "water bottle"
    frames = sorted(glob.glob("exp_*.jpg"))[:1]
    for f in frames:
        img = cv2.imread(f)
        cfg, dt = configure(img, target)
        if not cfg:
            continue
        t0 = time.time()
        cands = detect_candidates(img, cfg)
        print(f"  opencv found {len(cands)} candidate(s) in "
              f"{(time.time()-t0)*1000:.1f} ms")
        for c in cands[:2]:
            ok, vdt = verify(img, c["box"], target)
            print(f"    box={c['box']} area={c['area']:.0f} -> "
                  f"verify={ok} ({vdt:.2f}s)")
