import json
import re

import torch

import clip

OCC_ANCHORS = [
    "a human hand", "a person's arm", "fingers covering an object",
    "a sheet of paper", "a book", "a cloth", "fabric covering something",
    "someone's sleeve",
]
REP_ANCHORS = [
    "a cardboard box", "a plastic bottle", "a coffee mug", "a metal can",
    "a computer mouse", "an object sitting on a desk", "an empty desk surface",
    "a bottle of water",
]

OCCLUDER_WORDS = ("hand", "finger", "arm", "palm", "person", "glove",
                  "sleeve", "book", "paper", "cloth", "shadow", "sheet",
                  "hair", "phone", "wrist")


def kw_verdict(word):
    low = word.strip().lower()
    tokens = set(re.findall(r"[a-z]+", low))
    for t in list(tokens):
        if t.endswith("es") and len(t) > 3:
            tokens.add(t[:-2])
        if t.endswith("s") and len(t) > 2:
            tokens.add(t[:-1])
    return any(w in tokens for w in OCCLUDER_WORDS)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, _ = clip.load("ViT-B/32", device=device)
    cmodel.eval()
    with torch.no_grad():
        occ_emb = cmodel.encode_text(clip.tokenize(OCC_ANCHORS).to(device)).float()
        rep_emb = cmodel.encode_text(clip.tokenize(REP_ANCHORS).to(device)).float()
        occ_emb /= occ_emb.norm(dim=-1, keepdim=True)
        rep_emb /= rep_emb.norm(dim=-1, keepdim=True)

    def clip_text_verdict(word):
        with torch.no_grad():
            e = cmodel.encode_text(clip.tokenize([word]).to(device)).float()
            e /= e.norm(dim=-1, keepdim=True)
            so = (e @ occ_emb.T).max().item()
            sr = (e @ rep_emb.T).max().item()
        return so > sr

    items = []

    pr = json.load(open("probe_real.json"))
    for r in pr:
        for w in r["real_labels"]:
            items.append((w, True, f"real-occ:{r['scenario']}"))

    co = json.load(open("clip_occluder_classify.json"))
    for r in co:
        if r["source"].startswith("real:") and not r["truth_occluder"]:
            items.append((r["vlm_word"], False, "real-rep:" + r["source"][5:]))

    rows = []
    print(f"{'source':<26}{'word':<20}{'truth':>6}{'kw':>6}{'clip_text':>10}")
    print("-" * 70)
    for word, truth, src in items:
        kw = kw_verdict(word)
        ct = clip_text_verdict(word)
        rows.append(dict(source=src, word=word, truth_occluder=truth,
                         kw=kw, clip_text=ct))
        print(f"{src:<26}{word:<20}{('OCC' if truth else 'rep'):>6}"
              f"{('OCC' if kw else 'rep'):>6}{('OCC' if ct else 'rep'):>10}")

    def score(key):
        c = sum(1 for r in rows if r[key] == r["truth_occluder"])
        dang = sum(1 for r in rows if r[key] and not r["truth_occluder"])
        safe = sum(1 for r in rows if not r[key] and r["truth_occluder"])
        return c, len(rows), dang, safe

    print("\n" + "=" * 70)
    print(f"{'classifier':<12}{'correct':>10}{'dangerous':>12}{'safe':>10}")
    for key in ("kw", "clip_text"):
        c, n, dang, safe = score(key)
        print(f"{key:<12}{f'{c}/{n}':>10}{dang:>12}{safe:>10}")

    occ_rows = [r for r in rows if r["truth_occluder"]]
    kw_occ_correct = sum(1 for r in occ_rows if r["kw"])
    ct_occ_correct = sum(1 for r in occ_rows if r["clip_text"])
    print(f"\nOCCLUDER-direction only (n={len(occ_rows)}): "
          f"kw {kw_occ_correct}/{len(occ_rows)}  "
          f"clip_text {ct_occ_correct}/{len(occ_rows)}")

    json.dump(rows, open("clip_occluder_real2.json", "w"), indent=2)
    print("\nwrote clip_occluder_real2.json")


if __name__ == "__main__":
    main()
