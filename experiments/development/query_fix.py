from difflib import get_close_matches

COLOURS = ["red", "green", "blue", "yellow", "orange", "purple", "pink",
           "teal", "cyan", "white", "black", "gray", "grey", "brown",
           "silver", "gold", "clear", "transparent"]
ADJECTIVES = ["big", "small", "large", "little", "tiny", "huge", "shiny",
              "shiniest", "dark", "darkest", "bright", "brightest", "round",
              "roundest", "tall", "flat", "metal", "metallic", "plastic",
              "glass", "glossy", "biggest", "smallest", "wooden", "long"]
OBJECTS = ["cup", "mug", "bottle", "can", "keyboard", "mouse", "laptop",
           "monitor", "screen", "phone", "ball", "box", "book", "pen",
           "pencil", "scissors", "headphones", "speaker", "charger", "cable",
           "remote", "glasses", "wallet", "keys", "tape", "marker", "bowl",
           "plate", "spoon", "fork", "knife", "toy", "hat", "shoe", "towel",
           "tissue", "paper", "notebook", "tablet", "watch", "coin",
           "battery", "camera", "robot", "arm", "wire", "tool", "object",
           "thing", "water", "soda", "drink", "coffee", "tea", "bag",
           "backpack", "controller", "cap", "lid", "jar", "carton", "tumbler",
           "flask", "clock", "lamp", "light", "fan", "plant", "sticker"]

VOCAB = sorted(set(COLOURS + ADJECTIVES + OBJECTS))
CUTOFF = 0.78


def _fix_word(w):
    if w in VOCAB or len(w) < 3:
        return w
    for i in range(2, len(w) - 1):
        left, right = w[:i], w[i:]
        l_ok = left in VOCAB or bool(
            get_close_matches(left, VOCAB, n=1, cutoff=0.85))
        r_ok = right in VOCAB or bool(
            get_close_matches(right, VOCAB, n=1, cutoff=0.85))
        if l_ok and r_ok and len(left) >= 3 and len(right) >= 3:
            lm = left if left in VOCAB else get_close_matches(
                left, VOCAB, n=1, cutoff=0.85)[0]
            rm = right if right in VOCAB else get_close_matches(
                right, VOCAB, n=1, cutoff=0.85)[0]
            return f"{lm} {rm}"
    m = get_close_matches(w, VOCAB, n=1, cutoff=CUTOFF)
    return m[0] if m else w


def correct_query(query):
    words = query.lower().split()
    fixed = [_fix_word(w) for w in words]
    out = " ".join(fixed)
    return out, out != query.lower().strip()


if __name__ == "__main__":
    for q in ["redcup", "gren bottle", "kyboard", "watr bottle",
              "celsius can", "shinyest object", "bleu mug", "waterbottle",
              "owala bottle", "round ball"]:
        c, ch = correct_query(q)
        print(f"  {q!r:22} -> {c!r}" + ("   (corrected)" if ch else ""))
