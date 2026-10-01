
import time

import cv2
import numpy as np

BOTTLE_HSV = [(np.array([75, 35, 30]), np.array([110, 255, 255]))]

MIN_AREA = 3000
AREA_REF = 44104.0
MORPH_KERNEL = np.ones((5, 5), np.uint8)


def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in BOTTLE_HSV:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.erode(mask, MORPH_KERNEL, iterations=1)
    mask = cv2.dilate(mask, MORPH_KERNEL, iterations=2)
    return mask


def largest_contour(mask, min_area=MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    c = max(contours, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < min_area:
        return None, a
    return c, a


def shape_stats(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    ha = cv2.contourArea(hull)
    solidity = area / ha if ha > 0 else 0.0
    aspect = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0
    return {"solidity": round(solidity, 3), "aspect": round(aspect, 3),
            "extent": round(extent, 3), "area": area, "box": (x, y, w, h)}


def t1_hsv(frame, ctx=None):
    mask = green_mask(frame)
    c, area = largest_contour(mask)
    if c is None:
        return {"found": False, "box": None, "confidence": 0.0,
                "meta": {"area": area}}
    x, y, w, h = cv2.boundingRect(c)
    conf = min(1.0, area / AREA_REF)
    return {"found": True, "box": (x, y, w, h), "confidence": round(conf, 3),
            "meta": shape_stats(c)}


def t2_hsv_shape(frame, ctx=None):
    mask = green_mask(frame)
    c, area = largest_contour(mask)
    if c is None:
        return {"found": False, "box": None, "confidence": 0.0,
                "meta": {"area": area, "reason": "no_contour"}}
    st = shape_stats(c)
    reasons = []
    if st["aspect"] > 1.6:
        reasons.append("too_wide")
    if st["solidity"] < 0.70:
        reasons.append("low_solidity")
    if st["extent"] < 0.35:
        reasons.append("low_extent")
    ok = not reasons
    conf = min(1.0, area / AREA_REF) * (1.0 if ok else 0.25)
    return {"found": ok, "box": st["box"] if ok else None,
            "confidence": round(conf, 3),
            "meta": dict(st, rejected_for=reasons)}


_orb = cv2.ORB_create(nfeatures=1500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def t3_orb(frame, ctx):
    tmpl = ctx.get("template_gray")
    if tmpl is None:
        return {"found": False, "box": None, "confidence": 0.0,
                "meta": {"reason": "no_template"}}
    kp_t, des_t = ctx["template_orb"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = _orb.detectAndCompute(gray, None)
    if des_f is None or des_t is None or len(kp_f) < 2:
        return {"found": False, "box": None, "confidence": 0.0,
                "meta": {"reason": "no_descriptors",
                         "kp_frame": 0 if kp_f is None else len(kp_f)}}
    matches = _bf.knnMatch(des_t, des_f, k=2)
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    conf = min(1.0, len(good) / 40.0)
    box = None
    if len(good) >= 10:
        pts = np.float32([kp_f[m.trainIdx].pt for m in good])
        x, y, w, h = cv2.boundingRect(pts)
        box = (int(x), int(y), int(w), int(h))
    return {"found": len(good) >= 10, "box": box, "confidence": round(conf, 3),
            "meta": {"good_matches": len(good), "kp_frame": len(kp_f)}}


def t4_template(frame, ctx):
    tmpl = ctx.get("template_gray")
    if tmpl is None:
        return {"found": False, "box": None, "confidence": 0.0,
                "meta": {"reason": "no_template"}}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = (-1.0, None, None)
    for scale in (0.4, 0.55, 0.7, 0.85, 1.0, 1.2):
        th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
        if th < 12 or tw < 12 or th >= gray.shape[0] or tw >= gray.shape[1]:
            continue
        resized = cv2.resize(tmpl, (tw, th))
        res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv > best[0]:
            best = (maxv, (maxloc[0], maxloc[1], tw, th), scale)
    score, box, scale = best
    return {"found": score >= 0.5, "box": box if score >= 0.5 else None,
            "confidence": round(max(0.0, score), 3),
            "meta": {"best_score": round(score, 3), "best_scale": scale}}


def t5_vlm(frame, ctx):
    import llm_backend
    box = ctx.get("vlm_box")
    img = frame
    if box is not None:
        x, y, w, h = box
        px, py = int(w * 0.5), int(h * 0.5)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            img = crop
    prompt = ("Is there a green water bottle in this image? "
              "Respond with exactly one word: yes or no.")
    t0 = time.time()
    out = llm_backend.vision_chat(prompt, img, max_tokens=10)
    dt = time.time() - t0
    if out is None:
        return {"found": False, "box": box, "confidence": 0.0,
                "meta": {"raw": None, "latency_s": round(dt, 1),
                         "degenerate": True}}
    low = out.strip().lower()
    yes = low.startswith("yes") or (" yes" in low and " no" not in low)
    return {"found": yes, "box": box if yes else None,
            "confidence": 1.0 if yes else 0.0,
            "meta": {"raw": out, "latency_s": round(dt, 1)}}


def t6_ensemble(frame, ctx, precomputed=None):
    parts = precomputed or {
        "t1": t1_hsv(frame, ctx), "t2": t2_hsv_shape(frame, ctx),
        "t3": t3_orb(frame, ctx), "t4": t4_template(frame, ctx),
    }
    votes = [parts[k]["found"] for k in ("t1", "t2", "t3", "t4")]
    n_yes = sum(votes)
    box = None
    for k in ("t2", "t1", "t4", "t3"):
        if parts[k]["found"] and parts[k]["box"]:
            box = parts[k]["box"]
            break
    return {"found": n_yes >= 2, "box": box if n_yes >= 2 else None,
            "confidence": round(n_yes / 4.0, 3),
            "meta": {"votes": {k: parts[k]["found"] for k in parts},
                     "n_yes": n_yes}}


CLASSICAL = {"t1": t1_hsv, "t2": t2_hsv_shape, "t3": t3_orb, "t4": t4_template}


def build_context(template_bgr):
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = _orb.detectAndCompute(gray, None)
    return {"template_gray": gray, "template_orb": (kp, des),
            "template_kp_count": 0 if kp is None else len(kp)}
