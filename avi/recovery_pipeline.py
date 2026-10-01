import re
import time

import cv2
import numpy as np

CONFIRMED, OCCLUDED, DISPLACED, MISSING, AMBIGUOUS = (
    "CONFIRMED", "OCCLUDED", "DISPLACED", "MISSING", "AMBIGUOUS")

GLOBAL_CHANGE = 20.0
LOCAL_CHANGE = 25.0
EGO_MOTION_RATIO = 0.6
PERIPHERY_CHANGE = 48.0
OCCLUDER_IOU = 0.15
EDGE_STABILITY = 0.72
DECAY_FRAMES = 12


def _gray_small(f, size=(160, 120)):
    return cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2GRAY)


def _diff(a, b):
    return float(np.mean(cv2.absdiff(_gray_small(a), _gray_small(b))))


def _region_diff(a, b, box):
    x1, y1, x2, y2 = [max(0, v) for v in box]
    ra, rb = a[y1:y2, x1:x2], b[y1:y2, x1:x2]
    if ra.size == 0 or rb.size == 0 or ra.shape != rb.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(cv2.cvtColor(ra, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(rb, cv2.COLOR_BGR2GRAY))))


def _periphery_diff(a, b, box, pad_ratio=0.8):
    h, w = a.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)

    ga, gb = _gray_small(a), _gray_small(b)
    sh, sw = ga.shape
    ex1, ex2 = int(x1 * sw / w), int(x2 * sw / w)
    ey1, ey2 = int(y1 * sh / h), int(y2 * sh / h)
    d = cv2.absdiff(ga, gb).astype(np.float32)
    mask = np.ones_like(d, dtype=bool)
    mask[ey1:ey2, ex1:ex2] = False
    if mask.sum() < 32:
        return float(d.mean())
    return float(d[mask].mean())


def _edge_corr(a, b):
    ea = cv2.Canny(_gray_small(a), 60, 160).astype(np.float32).ravel()
    eb = cv2.Canny(_gray_small(b), 60, 160).astype(np.float32).ravel()
    if ea.std() < 1e-6 or eb.std() < 1e-6:
        return 1.0
    return float(np.corrcoef(ea, eb)[0, 1])


def _region_vs_surround(frame, box, pad_ratio=0.6):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return 1.0, 0.0

    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    ox1, oy1 = max(0, x1 - px), max(0, y1 - py)
    ox2, oy2 = min(w, x2 + px), min(h, y2 + py)

    inner = frame[y1:y2, x1:x2]
    outer = frame[oy1:oy2, ox1:ox2].copy()
    outer[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = 0
    ring = outer[outer.sum(axis=2) > 0]
    if ring.size == 0 or inner.size == 0:
        return 1.0, 0.0

    hi = cv2.calcHist([cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)], [0, 1],
                      None, [24, 24], [0, 180, 0, 256])
    ringimg = ring.reshape(-1, 1, 3).astype("uint8")
    hr = cv2.calcHist([cv2.cvtColor(ringimg, cv2.COLOR_BGR2HSV)], [0, 1],
                      None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hi, hi); cv2.normalize(hr, hr)
    sim = float(cv2.compareHist(hi, hr, cv2.HISTCMP_CORREL))

    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    uniformity = float(max(0.0, 1.0 - (g.std() / 64.0)))
    return sim, uniformity


SURROUND_SIMILAR = 0.67


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class RecoveryPipeline:

    def __init__(self, target, tracker=None, verifier=None,
                 use_detector=True, use_opencv=True, use_vlm=True,
                 use_state=True, decay_frames=DECAY_FRAMES,
                 surround_similar=None, use_region_probe=False,
                 probe_after=4, periphery_change=None, probe_verifier=None,
                 probe_every=8, probe_classifier=None, verify_every=8):
        self.target = target
        self.tracker = tracker
        self.verifier = verifier
        self.use_detector = use_detector
        self.use_opencv = use_opencv
        self.use_vlm = use_vlm
        self.use_state = use_state
        self.decay_frames = decay_frames
        self.surround_similar = (SURROUND_SIMILAR if surround_similar is None
                                 else surround_similar)
        self.use_region_probe = use_region_probe
        self.probe_after = probe_after
        self.probe_verifier = probe_verifier
        self.probe_every = probe_every
        self.verify_every = verify_every
        self.probe_classifier = probe_classifier
        self.periphery_change = (PERIPHERY_CHANGE if periphery_change is None
                                 else periphery_change)
        self.reset()

    def notify_self_motion(self):
        self._self_moved = True
        self.clean_ref = None

    def reset(self):
        self._probe_at = None
        self._probe_verdict = None
        self._reject_streak = 0
        self._self_moved = False
        self.status = MISSING
        self.box = None
        self.conf = 0.0
        self.frames_since_seen = 0
        self.prev = None
        self.clean_ref = None
        self.vlm_calls = 0
        self.log = []

    def _vlm_whole_frame(self, frame):
        if self.verifier is None:
            return None
        self.vlm_calls += 1
        ans = self.verifier(
            f"Answer yes or no only. Is there a {self.target} in this image?",
            frame)
        if ans and ans.strip().lower().startswith("y"):
            h, w = frame.shape[:2]
            return (1.0, [0, 0, w, h])
        return None

    def _detect(self, frame):
        if not self.use_detector:
            return self._vlm_whole_frame(frame), []
        if self.tracker is None:
            return None, []
        res = self.tracker.model.predict(frame, conf=0.10, verbose=False)[0]
        target_hit, others = None, []
        for b in res.boxes:
            name = self.tracker.model.names[int(b.cls.item())]
            c = float(b.conf.item())
            xy = [int(v) for v in b.xyxy[0].tolist()]
            if name == self.target:
                if target_hit is None or c > target_hit[0]:
                    target_hit = (c, xy)
            elif c >= 0.25:
                others.append((name, c, xy))
        return target_hit, others

    def _diagnose(self, frame, others):
        ev = {}
        if self._self_moved:
            self._self_moved = False
            return AMBIGUOUS, {"reason": "self-commanded motion, not a disruption"}
        if not self.use_opencv or self.clean_ref is None or self.box is None:
            return MISSING, {"reason": "no opencv / no reference"}

        g = _diff(frame, self.clean_ref)
        l = _region_diff(frame, self.clean_ref, self.box)
        e = _edge_corr(frame, self.clean_ref)
        p = _periphery_diff(frame, self.clean_ref, self.box)
        ev.update(global_diff=round(g, 2), local_diff=round(l, 2),
                  edge_corr=round(e, 3), periphery_diff=round(p, 2))

        if (g >= GLOBAL_CHANGE and p >= self.periphery_change
                and (l <= 1e-6 or g / max(l, 1e-6) >= EGO_MOTION_RATIO)):
            if e < EDGE_STABILITY:
                return DISPLACED, ev

        if g >= GLOBAL_CHANGE and e >= EDGE_STABILITY:
            ev["reason"] = "lighting: structure preserved"
            return AMBIGUOUS, ev

        for name, c, xy in others:
            iou = _iou(self.box, xy)
            if iou >= OCCLUDER_IOU:
                ev.update(occluder=name, occluder_iou=round(iou, 3))
                return OCCLUDED, ev

        if l >= LOCAL_CHANGE:
            sim, uni = _region_vs_surround(frame, self.box)
            ev.update(surround_sim=round(sim, 3), uniformity=round(uni, 3))
            if sim >= self.surround_similar:
                ev["reason"] = "region now matches background -> removed"
                return MISSING, ev
            ev["reason"] = "region differs from background -> something covers it"
            return OCCLUDED, ev

        ev["reason"] = "region unchanged; detector miss"
        return AMBIGUOUS, ev

    def _verify(self, frame, box):
        if not self.use_vlm or self.verifier is None:
            return True, "no-vlm"
        x1, y1, x2, y2 = [max(0, v) for v in box]
        pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y),
                     max(0, x1 - pad_x):min(w, x2 + pad_x)]
        if crop.size == 0:
            return False, "empty crop"
        self.vlm_calls += 1
        ans = self.verifier(
            f"Answer yes or no only. Is the main object in this image a "
            f"{self.target}?", crop)
        ok = bool(ans) and ans.strip().lower().startswith("y")
        return ok, (ans or "").strip()[:40]

    OCCLUDER_WORDS = ("hand", "finger", "arm", "palm", "person", "glove",
                      "sleeve", "book", "paper", "cloth", "shadow", "sheet",
                      "hair", "phone", "wrist")

    def probe_region(self, frame):
        v = self.probe_verifier
        if v is None or self.box is None:
            return None, None
        x1, y1, x2, y2 = [max(0, v) for v in self.box]
        h, w = frame.shape[:2]
        px, py = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        crop = frame[max(0, y1 - py):min(h, y2 + py),
                     max(0, x1 - px):min(w, x2 + px)]
        if crop.size == 0:
            return None, None
        self.vlm_calls += 1
        ans = v("What is the main object in this image? Answer with one or "
                "two words only.", crop)
        if not ans:
            return None, None
        low = ans.strip().lower()
        tokens = set(re.findall(r"[a-z]+", low))
        for t in list(tokens):
            if t.endswith("es") and len(t) > 3:
                tokens.add(t[:-2])
            if t.endswith("s") and len(t) > 2:
                tokens.add(t[:-1])
        if self.probe_classifier is not None:
            return low[:30], bool(self.probe_classifier(low))
        return low[:30], any(word in tokens for word in self.OCCLUDER_WORDS)

    def step(self, frame):
        t0 = time.time()
        hit, others = self._detect(frame)
        rec = {"detected": hit is not None,
               "conf": round(hit[0], 3) if hit else 0.0,
               "vlm": None, "evidence": {}}

        if not self.use_state:
            if hit and self.use_vlm and self.use_detector:
                ok, ans = self._verify(frame, hit[1])
                rec["vlm"] = ans
                hit = hit if ok else None
            self.status = CONFIRMED if hit else MISSING
            if hit:
                self.box, self.conf = hit[1], hit[0]
            rec.update(status=self.status, action="none",
                       elapsed=round(time.time() - t0, 3))
            self.log.append(rec)
            return rec

        if hit:
            self.frames_since_seen = 0
            self._probe_at = None
            self._probe_verdict = None
            if self.status in (CONFIRMED,):
                self.box, self.conf = hit[1], hit[0]
                self.clean_ref = frame.copy()
                action = "track"
            else:
                due = self._reject_streak % self.verify_every == 0
                if due:
                    ok, ans = self._verify(frame, hit[1])
                    rec["vlm"] = ans
                else:
                    ok, ans = False, None
                if ok:
                    self.status = CONFIRMED
                    self.box, self.conf = hit[1], hit[0]
                    self.clean_ref = frame.copy()
                    self._reject_streak = 0
                    action = "reacquired+verified"
                else:
                    self.status = AMBIGUOUS
                    self._reject_streak += 1
                    action = ("rejected: not the target" if due
                              else "rejected: throttled, no VLM call")
        else:
            self._reject_streak = 0
            self.frames_since_seen += 1
            status, ev = self._diagnose(frame, others)
            rec["evidence"] = ev
            covered = False
            if (status == OCCLUDED and self.use_region_probe
                    and self.frames_since_seen > self.probe_after):
                due = (self._probe_at is None
                       or self.frames_since_seen - self._probe_at
                       >= self.probe_every)
                if due:
                    self._probe_at = self.frames_since_seen
                    self._probe_verdict = self.probe_region(frame)
                label, is_occ = self._probe_verdict or (None, None)
                if label is not None and not is_occ:
                    status = MISSING
                    ev["reason"] = (f"region holds {label!r}, not an occluder "
                                    f"-> target was replaced")
                elif label is not None:
                    covered = True
                    ev["reason"] = f"region holds {label!r} -> still covered"

            if (status == OCCLUDED and not covered
                    and self.frames_since_seen > self.decay_frames):
                status = MISSING
                ev["reason"] = f"occlusion decayed after {self.decay_frames} frames"
            self.status = status
            action = {OCCLUDED: "wait",
                      DISPLACED: "re-localise",
                      MISSING: "search",
                      AMBIGUOUS: "re-verify in place"}.get(status, "none")

        rec.update(status=self.status, action=action,
                   frames_since_seen=self.frames_since_seen,
                   elapsed=round(time.time() - t0, 3))
        self.prev = frame
        self.log.append(rec)
        return rec

    SEARCH_DIRS = ("left", "right", "up", "down")

    def search_direction(self, frame, tried=None):
        if not self.use_vlm or self.verifier is None:
            return None
        tried = tried or {}
        swept = ", ".join(f"{d} ({n}x)" for d, n in tried.items() if n) or "none"
        self.vlm_calls += 1
        ans = self.verifier(
            f"A camera is looking for a {self.target}, which is NOT in view. "
            f"Based on what you can see, which way should the camera turn to "
            f"find it? Already tried: {swept}. "
            f"Answer with exactly one word: left, right, up, or down.",
            frame)
        if not ans:
            return None
        low = ans.strip().lower()
        for d in self.SEARCH_DIRS:
            if d in low:
                return d

        if not getattr(self, "_warned_search_dead", False):
            self._warned_search_dead = True
            print(f"[search_direction] verifier answered {ans!r} -- no "
                  f"direction parsed. See assignment3b_offline.py; with the "
                  f"fast yes/no verifier this path never yields a direction.")
        return None

    def believes_present(self):
        return self.status in (CONFIRMED, OCCLUDED)
