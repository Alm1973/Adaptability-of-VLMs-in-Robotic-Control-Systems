import numpy as np
import cv2


def spectral_residual_saliency(gray, small=64):
    h, w = gray.shape
    g = cv2.resize(gray, (small, small)).astype(np.float32)
    f = np.fft.fft2(g)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    avg = cv2.blur(log_amp, (3, 3))
    residual = log_amp - avg
    recon = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = np.abs(recon) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=2.5)
    sal = cv2.resize(sal, (w, h))
    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()
    return sal


def _merge_boxes(boxes, iou_gap=0.0):
    merged = True
    boxes = list(boxes)
    while merged:
        merged = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            x2, y2 = x + w, y + h
            grew = True
            while grew:
                grew = False
                rem = []
                for (bx, by, bw, bh) in boxes:
                    bx2, by2 = bx + bw, by + bh
                    if (bx <= x2 and bx2 >= x and by <= y2 and by2 >= y):
                        x, y = min(x, bx), min(y, by)
                        x2, y2 = max(x2, bx2), max(y2, by2)
                        grew = True
                        merged = True
                    else:
                        rem.append((bx, by, bw, bh))
                boxes = rem
            out.append((x, y, x2 - x, y2 - y))
        boxes = out
    return boxes


def propose_regions(img_bgr, max_regions=3, min_frac=0.03, pad_frac=0.10,
                    min_crop_frac=0.30):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sal = spectral_residual_saliency(gray)

    thr = sal.mean() + sal.std()
    mask = (sal >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw * bh < min_frac * w * h:
            continue
        region_sal = float(sal[y:y + bh, x:x + bw].sum())
        cand.append((region_sal, (x, y, bw, bh)))
    cand.sort(key=lambda t: -t[0])
    boxes = [b for _, b in cand[:max_regions * 2]]
    boxes = _merge_boxes(boxes)
    scored = []
    for (x, y, bw, bh) in boxes:
        scored.append((float(sal[y:y + bh, x:x + bw].sum()), (x, y, bw, bh)))
    scored.sort(key=lambda t: -t[0])
    boxes = [b for _, b in scored[:max_regions]]

    out = []
    px, py = pad_frac * w, pad_frac * h
    minw, minh = min_crop_frac * w, min_crop_frac * h
    for (x, y, bw, bh) in boxes:
        x0, y0, x1, y1 = x - px, y - py, x + bw + px, y + bh + py
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cw, ch = max(x1 - x0, minw), max(y1 - y0, minh)
        x0, x1 = cx - cw / 2, cx + cw / 2
        y0, y1 = cy - ch / 2, cy + ch / 2
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        out.append((y0 / h, y1 / h, x0 / w, x1 / w))
    return out


if __name__ == "__main__":
    import sys
    from heldout_gt import HELDOUT_GT

    frames = sys.argv[1:] or sorted(HELDOUT_GT)
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            print(f"{f}: unreadable")
            continue
        h, w = img.shape[:2]
        regions = propose_regions(img)
        print(f"\n{f} ({w}x{h}) -> {len(regions)} regions")
        viz = img.copy()
        for i, (r0, r1, c0, c1) in enumerate(regions):
            x0, y0 = int(c0 * w), int(r0 * h)
            x1, y1 = int(c1 * w), int(r1 * h)
            cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 0, 255), 4)
            cv2.putText(viz, str(i), (x0 + 5, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            frac = (r1 - r0) * (c1 - c0)
            print(f"  region {i}: r[{r0:.2f},{r1:.2f}] c[{c0:.2f},{c1:.2f}]"
                  f"  covers {frac*100:.0f}% of frame")
            crop = img[y0:y1, x0:x1]
            cv2.imwrite(f"_propcrop_{f.split('_')[0]}_{f.split('.')[0][-4:]}_{i}.jpg", crop)
        out = f"_proposer_{f.replace('.jpg','')}.jpg"
        cv2.imwrite(out, viz)
        print(f"  wrote {out}")
