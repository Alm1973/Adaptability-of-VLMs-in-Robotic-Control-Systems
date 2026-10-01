import json
import statistics as st
import time

import cv2

import llm_backend
from heldout_gt import HELDOUT_GT
from open_vocab_detect import (LIST_PROMPT, MAX_TOKENS, VOTE_MIN_AGREE,
                               VOTE_SAMPLES, _dedupe, _parse_items, is_present)

REPEATS = 3
QUERIES = ["laptop", "keyboard", "computer mouse", "water bottle"]

CONFUSABLE = {
    "computer mouse": "headphones",
    "water bottle": "a drink can",
    "laptop": "a computer monitor",
}

CHOICE_PROMPT = ("Look closely at this image. Is the object in question "
                 "a {a} or {b}? Consider shape and detail carefully. "
                 "Answer with ONLY one of these two words: "
                 "'{a_word}' or '{b_word}'.")


def one_list(img):
    txt = llm_backend.vision_chat(LIST_PROMPT, img, max_tokens=MAX_TOKENS)
    return _dedupe(_parse_items(txt)) if txt else []


def listing_votes(img, samples=VOTE_SAMPLES):
    lists = [one_list(img) for _ in range(samples)]
    return {q: sum(1 for l in lists if is_present(q, l)) >= VOTE_MIN_AGREE
            for q in QUERIES}


def disambiguate(img, query):
    rival = CONFUSABLE[query]
    a_word = query.split()[-1]
    b_word = rival.split()[-1]
    prompt = CHOICE_PROMPT.format(a=query, b=rival,
                                  a_word=a_word, b_word=b_word)
    out = llm_backend.vision_chat(prompt, img, max_tokens=12)
    if not out:
        return True
    t = out.strip().lower()
    if b_word in t and a_word not in t:
        return False
    return True


def main():
    frames = list(HELDOUT_GT)
    imgs = {f: cv2.imread(f) for f in frames}
    if any(v is None for v in imgs.values()):
        print("missing frames")
        return
    print(f"DISAMBIGUATION: {len(frames)} held-out frames x {REPEATS} repeats")
    print(f"pairs: {CONFUSABLE}\n")

    runs = {"listing_only": [], "listing_disambig": []}
    flips = []
    for rep in range(1, REPEATS + 1):
        base_c = base_t = dis_c = 0
        extra_calls = 0
        t0 = time.time()
        for f in frames:
            preds = listing_votes(imgs[f])
            dis = dict(preds)
            for q in QUERIES:
                if preds[q] and q in CONFUSABLE:
                    extra_calls += 1
                    keep = disambiguate(imgs[f], q)
                    if not keep:
                        dis[q] = False
                        flips.append({"rep": rep, "frame": f, "query": q,
                                      "gt": HELDOUT_GT[f][q]})
            for q in QUERIES:
                gt = HELDOUT_GT[f][q]
                base_t += 1
                base_c += int(preds[q]) == gt
                dis_c += int(dis[q]) == gt
        dt = time.time() - t0
        runs["listing_only"].append(round(base_c / base_t, 3))
        runs["listing_disambig"].append(round(dis_c / base_t, 3))
        print(f"  repeat {rep}: listing {base_c}/{base_t} "
              f"({base_c/base_t:.3f})  +disambig {dis_c}/{base_t} "
              f"({dis_c/base_t:.3f})  extra_calls {extra_calls}  {dt:.0f}s")
        json.dump({"runs": runs, "flips": flips},
                  open("disambiguation_test.json", "w"), indent=2)

    llm_backend.shutdown()
    good = [f for f in flips if f["gt"] == 0]
    bad = [f for f in flips if f["gt"] == 1]
    print("\n===== RESULT =====")
    print(f"listing_only     mean acc {st.mean(runs['listing_only']):.3f} "
          f"{runs['listing_only']}")
    print(f"listing+disambig mean acc "
          f"{st.mean(runs['listing_disambig']):.3f} "
          f"{runs['listing_disambig']}")
    print(f"\nflips made: {len(flips)}  "
          f"-> removed {len(good)} FALSE positives, "
          f"destroyed {len(bad)} TRUE positives")
    for fl in flips:
        tag = "good" if fl["gt"] == 0 else "BAD "
        print(f"  {tag} rep{fl['rep']} {fl['frame'][:12]} {fl['query']}")
    print("\nSHIP RULE: adopt only if mean accuracy improves AND "
          "true positives lost <= false positives removed.")


if __name__ == "__main__":
    main()
