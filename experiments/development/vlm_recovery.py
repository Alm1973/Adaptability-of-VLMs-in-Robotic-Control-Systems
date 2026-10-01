
import cv2

import llm_backend
from shape_signals import compute_shape_signals, classify_from_signals
from config import VERIFICATION_CROP_PADDING_RATIO

BASE_PROMPT = (
    "You are controlling a camera-mounted robotic arm searching for a red "
    "cup. This image is a map of what the arm has already looked at: gray "
    "squares are unexplored positions, real photos are positions the arm "
    "has already photographed.\n\n"
    "If you can clearly see the red cup in one of the photographed tiles, "
    "respond with exactly one word: object_found\n\n"
    "Otherwise, look at where the gray (unexplored) squares are relative "
    "to the photographed squares, and respond with exactly one of these "
    "words for the general direction with the most unexplored territory: "
    "scan_left, scan_right, tilt_up, tilt_down.\n\n"
    "If the cup is not visible anywhere in the image, pick a direction you "
    "have NOT tried yet based on the scan history below.\n\n"
    "Respond with only that single word, nothing else."
)

VALID_ACTIONS = ["scan_left", "scan_right", "tilt_up", "tilt_down", "object_found"]

DIRECTION_HINT_PHRASES = {
    "scan_left": "toward the left",
    "scan_right": "toward the right",
    "tilt_up": "upward",
    "tilt_down": "downward",
}

VERIFICATION_PROMPT = (
    "This image is a close-up crop of a red object detected by a robotic "
    "arm's camera. Is this object a cup, or is it something else such as a "
    "hand, arm, or other skin-toned object? Respond with exactly one word: "
    "cup or other."
)

VERIFICATION_VALID_WORDS = ["cup", "other"]

MODEL_NAME = "qwen2.5vl:3b"

OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 20,
}

MAX_TOKENS = 20


def extract_action(raw_text):
    text = raw_text.lower()
    found = [action for action in VALID_ACTIONS if action in text]
    if not found:
        return None
    found.sort(key=lambda a: text.rfind(a))
    return found[-1]


def extract_verification(raw_text):
    text = raw_text.lower()
    found = [w for w in VERIFICATION_VALID_WORDS if w in text]
    if not found:
        return None
    found.sort(key=lambda w: text.rfind(w))
    return found[-1]


def _crop_with_padding(frame, box):
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]
    pad_x = int(w * VERIFICATION_CROP_PADDING_RATIO)
    pad_y = int(h * VERIFICATION_CROP_PADDING_RATIO)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame_w, x + w + pad_x)
    y1 = min(frame_h, y + h + pad_y)
    return frame[y0:y1, x0:x1]


def get_recovery_action(mosaic, scan_history=None, last_known_direction=None):
    prompt = BASE_PROMPT

    phrase = DIRECTION_HINT_PHRASES.get(last_known_direction)
    if phrase:
        print(f"[VLM] Using last-known-direction hint: {last_known_direction}")
        prompt += (
            f"\n\nHint: the object was last seen moving {phrase} before "
            f"tracking was lost, so you may want to start your search in "
            f"that direction. This is only a hint, not a rule -- base your "
            f"actual decision on the mosaic and the scan history below."
        )

    if scan_history:
        prompt += f"\n\nScan history this session: {scan_history}"

    raw_text = llm_backend.vision_chat(prompt, mosaic, max_tokens=MAX_TOKENS)
    if raw_text is None:
        return None

    action = extract_action(raw_text)
    if action is None:
        print(f"[VLM][DEGENERATE] full_len={len(raw_text)} full_text={raw_text!r}")
        return None
    return action


def get_verification(frame, box):
    crop = _crop_with_padding(frame, box)
    if crop.size == 0:
        print("[VERIFY] Empty crop, skipping")
        return None

    raw_text = llm_backend.vision_chat(VERIFICATION_PROMPT, crop, max_tokens=MAX_TOKENS)
    if raw_text is None:
        return None

    outcome = extract_verification(raw_text)
    if outcome is None:
        print(f"[VERIFY][DEGENERATE] full_len={len(raw_text)} full_text={raw_text!r}")
        return None
    return outcome


def get_verification_hybrid(frame, box, contour):
    if contour is not None:
        signals = compute_shape_signals(contour)
        gate = classify_from_signals(signals)
        if gate is not None:
            print(f"[VERIFY] Classical gate rejected: {gate} (signals={signals})")
            return gate
        print(f"[VERIFY] Classical gate inconclusive (signals={signals}), "
              f"deferring to VLM")
    else:
        print("[VERIFY] No contour available, going straight to VLM")

    return get_verification(frame, box)


def warm_up():
    print("[VLM] Warming up inference backend...")
    if llm_backend.ensure_running():
        print("[VLM] Backend ready (local llama-server, GPU).")
    else:
        print("[VLM] Local llama-server unavailable -- calls will use the "
              "ollama-CPU fallback (slow).")


def force_reload_model():
    print("[VLM] Forcing backend restart due to repeated degenerate output...")
    llm_backend.restart()
