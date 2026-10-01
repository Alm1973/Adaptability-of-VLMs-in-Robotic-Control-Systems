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

_cache = {}


def make_clip_classifier(device=None):
    if "fn" in _cache:
        return _cache["fn"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    with torch.no_grad():
        occ_emb = model.encode_text(clip.tokenize(OCC_ANCHORS).to(device)).float()
        rep_emb = model.encode_text(clip.tokenize(REP_ANCHORS).to(device)).float()
        occ_emb /= occ_emb.norm(dim=-1, keepdim=True)
        rep_emb /= rep_emb.norm(dim=-1, keepdim=True)

    def classify(word):
        if not word:
            return False
        with torch.no_grad():
            e = model.encode_text(clip.tokenize([word]).to(device)).float()
            e /= e.norm(dim=-1, keepdim=True)
            so = (e @ occ_emb.T).max().item()
            sr = (e @ rep_emb.T).max().item()
        return so > sr

    _cache["fn"] = classify
    return classify
