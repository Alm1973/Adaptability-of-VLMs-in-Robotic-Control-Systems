import numpy as np

from recovery_pipeline import RecoveryPipeline, OCCLUDED


class FakeTracker:

    def __init__(self):
        self.visible = True

        class M:
            names = {0: "red cup"}

            def predict(self_inner, frame, conf=0.1, verbose=False):
                class B:
                    boxes = []
                return [B()]
        self.model = M()


def main():
    calls = {"n": 0}

    def fake_probe(prompt, crop):
        calls["n"] += 1
        return "a hand"

    pipe = RecoveryPipeline("red cup", tracker=None, verifier=None,
                            use_detector=False, use_opencv=True,
                            use_vlm=False, use_state=True,
                            use_region_probe=True, probe_verifier=fake_probe,
                            probe_after=2, probe_every=4)

    frame = (np.random.RandomState(0)
             .randint(0, 255, (240, 320, 3)).astype(np.uint8))
    pipe.box = [100, 80, 200, 180]
    pipe.clean_ref = frame.copy()

    ok = True

    pipe.status = OCCLUDED
    for i in range(12):
        pipe.frames_since_seen = i
        pipe._maybe_probe_for_test = None
        if i > pipe.probe_after:
            due = (pipe._probe_at is None
                   or pipe.frames_since_seen - pipe._probe_at
                   >= pipe.probe_every)
            if due:
                pipe._probe_at = pipe.frames_since_seen
                pipe._probe_verdict = pipe.probe_region(frame)
    first = calls["n"]
    print(f"occlusion 1: probe fired {first}x")
    if first < 1:
        print("  !! probe never fired on the first occlusion")
        ok = False

    pipe.frames_since_seen = 0
    pipe._probe_at = None
    pipe._probe_verdict = None

    before = calls["n"]
    for i in range(12):
        pipe.frames_since_seen = i
        if i > pipe.probe_after:
            due = (pipe._probe_at is None
                   or pipe.frames_since_seen - pipe._probe_at
                   >= pipe.probe_every)
            if due:
                pipe._probe_at = pipe.frames_since_seen
                pipe._probe_verdict = pipe.probe_region(frame)
    second = calls["n"] - before
    print(f"occlusion 2: probe fired {second}x")
    if second < 1:
        print("  !! probe did NOT fire on the second occlusion -- the stale "
              "_probe_at\n     regression is back. It fails SILENTLY: no "
              "error, belief just stops\n     being held on every occlusion "
              "after the first.")
        ok = False

    pipe.reset()
    if pipe._probe_at is not None or pipe._probe_verdict is not None:
        print("  !! reset() left probe state behind")
        ok = False
    else:
        print("reset() clears probe state")

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
