import numpy as np
from recovery_pipeline import RecoveryPipeline, CONFIRMED, AMBIGUOUS

frame = np.zeros((100, 100, 3), dtype=np.uint8)
HIT = (0.4, [10, 10, 40, 40])


def test_persistent_rejection_is_throttled():
    calls = {"n": 0}

    def always_no(prompt, crop):
        calls["n"] += 1
        return "no"

    pipe = RecoveryPipeline("red cup", tracker=None, verifier=always_no,
                             use_detector=True, use_vlm=True, use_state=True,
                             verify_every=8)
    pipe._detect = lambda f: (HIT, [])

    N = 40
    statuses = [pipe.step(frame)["status"] for _ in range(N)]

    assert calls["n"] == (N + pipe.verify_every - 1) // pipe.verify_every, (
        f"expected ceil({N}/{pipe.verify_every}) calls, got {calls['n']}")
    assert all(s == AMBIGUOUS for s in statuses), (
        "throttle must not change belief -- every frame should still read "
        "AMBIGUOUS, same as the unthrottled pipeline would report")
    print(f"PASS persistent_rejection_is_throttled: {calls['n']}/{N} calls, "
          f"all AMBIGUOUS")


def test_streak_resets_on_confirm_and_on_lost_detection():
    calls = {"n": 0, "next": "no"}

    def scripted(prompt, crop):
        calls["n"] += 1
        return calls["next"]

    pipe = RecoveryPipeline("red cup", tracker=None, verifier=scripted,
                             use_detector=True, use_vlm=True, use_state=True,
                             verify_every=8)
    pipe._detect = lambda f: (HIT, [])

    pipe.step(frame)
    assert pipe.status == AMBIGUOUS and pipe._reject_streak == 1
    assert calls["n"] == 1
    pipe._reject_streak = 8
    calls["next"] = "yes"
    r = pipe.step(frame)
    assert r["status"] == CONFIRMED, r["status"]
    assert pipe._reject_streak == 0, "streak must reset on CONFIRMED"
    assert calls["n"] == 2
    print("PASS streak_resets_on_confirm")

    pipe.reset()
    calls["n"] = 0
    calls["next"] = "no"
    pipe._detect = lambda f: (HIT, [])
    pipe.step(frame)
    assert calls["n"] == 1 and pipe._reject_streak == 1
    pipe._detect = lambda f: (None, [])
    pipe.step(frame)
    assert pipe._reject_streak == 0, "streak must reset when detection is lost"
    pipe._detect = lambda f: (HIT, [])
    pipe.step(frame)
    assert calls["n"] == 2, (
        "a fresh detection after a gap must verify immediately, not inherit "
        "the throttle budget from an unrelated earlier rejection streak")
    print("PASS streak_resets_on_lost_detection")


if __name__ == "__main__":
    test_persistent_rejection_is_throttled()
    test_streak_resets_on_confirm_and_on_lost_detection()
    print("ALL PASS")
