import re

import llm_backend

LIST_PROMPT = ("List every distinct physical object you can clearly see in "
               "this image. Reply with ONLY a comma-separated list of short "
               "object names, nothing else.")

MAX_ITEMS = 25
MAX_TOKENS = 140

SYNONYMS = {
    "water bottle": ["water bottle", "bottle", "flask", "tumbler", "thermos",
                     "canteen"],
    "keyboard": ["keyboard", "keypad"],
    "computer mouse": ["computer mouse", "mouse"],
    "laptop": ["laptop", "notebook computer", "macbook"],
    "monitor": ["monitor", "screen", "display"],
    "cup": ["cup", "mug", "glass"],
    "can": ["can", "soda can", "drink can"],
    "phone": ["phone", "smartphone", "cellphone", "mobile"],
    "headphones": ["headphones", "headset", "earphones"],
    "banana": ["banana"],
}
BLOCK_SUBSTR = {
    "computer mouse": ["mousepad", "mouse pad", "mouse mat"],
    "keyboard": ["keyboard tray"],
}


def _parse_items(text):
    if not text:
        return []
    t = re.sub(r"^[^:]{0,40}:", "", text.strip())
    out = []
    for p in re.split(r"[,\n;]+", t):
        p = re.sub(r"^\s*[-*\d.)]+\s*", "", p).strip()
        if p:
            out.append(p)
    return out


def _dedupe(items):
    seen, out = set(), []
    for it in items:
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= MAX_ITEMS:
            break
    return out


VOTE_SAMPLES = 3
VOTE_MIN_AGREE = 2


def list_objects(image_bgr, samples=1):
    out = []
    for _ in range(max(1, samples)):
        txt = llm_backend.vision_chat(LIST_PROMPT, image_bgr,
                                      max_tokens=MAX_TOKENS)
        out.append(_dedupe(_parse_items(txt)) if txt else [])
    return out


_STOPWORDS = {"a", "an", "the", "some", "my", "of", "with", "that", "this"}


def _matches(query, item):
    s = item.strip().lower()
    q = query.strip().lower()
    if any(b in s for b in BLOCK_SUBSTR.get(q, [])):
        return False

    for syn in SYNONYMS.get(q, [q]):
        if re.search(rf"\b{re.escape(syn)}\b", s):
            return True

    if len(s) > 2 and re.search(rf"\b{re.escape(s)}\b", q):
        return True

    qwords = [w for w in re.findall(r"[a-z]+", q) if w not in _STOPWORDS]
    if len(qwords) > 1:
        head = qwords[-1]
        if len(head) > 2 and re.search(rf"\b{re.escape(head)}\b", s):
            return True
    return False


def is_present(query, items):
    return any(_matches(query, it) for it in items)


def is_present_voted(query, sample_lists, min_agree=VOTE_MIN_AGREE):
    hits = sum(1 for items in sample_lists if is_present(query, items))
    return hits >= min_agree


def detect(image_bgr, queries, samples=VOTE_SAMPLES,
           min_agree=VOTE_MIN_AGREE):
    if samples <= 1:
        items = list_objects(image_bgr, samples=1)[0]
        return {q: is_present(q, items) for q in queries}, [items]

    lists = list_objects(image_bgr, samples=samples - 1)
    counts = {q: sum(1 for l in lists if is_present(q, l)) for q in queries}
    unanimous = all(c == 0 or c == len(lists) for c in counts.values())
    if not unanimous:
        lists += list_objects(image_bgr, samples=1)
        counts = {q: sum(1 for l in lists if is_present(q, l))
                  for q in queries}
    return ({q: counts[q] >= min_agree for q in queries}, lists)


CROPS = {
    "TL":     (0.00, 0.60, 0.00, 0.60),
    "TR":     (0.00, 0.60, 0.40, 1.00),
    "BL":     (0.40, 1.00, 0.00, 0.60),
    "BR":     (0.40, 1.00, 0.40, 1.00),
    "center": (0.20, 0.80, 0.25, 0.75),
}


def _crop(image_bgr, box):
    h, w = image_bgr.shape[:2]
    r0, r1, c0, c1 = box
    return image_bgr[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)]


def list_objects_multiview(image_bgr, whole_samples=1):
    views = {}
    if whole_samples <= 1:
        views["whole"] = list_objects(image_bgr, samples=1)[0]
    else:
        for i, lst in enumerate(list_objects(image_bgr, samples=whole_samples)):
            views[f"whole{i}"] = lst
    for name, box in CROPS.items():
        views[name] = list_objects(_crop(image_bgr, box), samples=1)[0]
    return views


def detect_multiview(image_bgr, queries, min_views=2, whole_samples=1):
    views = list_objects_multiview(image_bgr, whole_samples=whole_samples)
    view_lists = list(views.values())
    results = {q: sum(1 for l in view_lists if is_present(q, l)) >= min_views
               for q in queries}
    return results, views


SCENE_DIFF_THRESHOLD = 20.0
_DIFF_SIZE = (160, 120)


def scene_diff(a, b):
    import cv2
    ga = cv2.cvtColor(cv2.resize(a, _DIFF_SIZE), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, _DIFF_SIZE), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


class SceneGatedDetector:

    def __init__(self, queries, threshold=SCENE_DIFF_THRESHOLD,
                 samples=VOTE_SAMPLES):
        self.queries = list(queries)
        self.threshold = threshold
        self.samples = samples
        self._last_frame = None
        self._last_result = None
        self._last_lists = None
        self.stats = {"detections": 0, "skips": 0}

    def invalidate(self):
        self._last_frame = None

    def detect(self, image_bgr):
        if self._last_frame is not None:
            if scene_diff(self._last_frame, image_bgr) < self.threshold:
                self.stats["skips"] += 1
                return self._last_result, False
        res, lists = detect(image_bgr, self.queries, samples=self.samples)
        self._last_frame = image_bgr.copy()
        self._last_result = res
        self._last_lists = lists
        self.stats["detections"] += 1
        return res, True


if __name__ == "__main__":
    import glob
    import sys

    import cv2

    paths = sys.argv[1:] or sorted(glob.glob("exp_*.jpg"))[:3]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: unreadable")
            continue
        res, lists = detect(img, ["water bottle", "keyboard",
                                  "computer mouse", "laptop", "banana"])
        print(f"\n{p}")
        for i, l in enumerate(lists, 1):
            print(f"  sample {i}: {l}")
        print(f"  -> {res}")
