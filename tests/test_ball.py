import numpy as np

from vbtrack.ball import BallCandidate, BallTracker


def cand(x, y, conf=0.6):
    return BallCandidate(x, y, 20, 20, conf)


def test_acquires_highest_confidence():
    bt = BallTracker()
    s = bt.step([cand(100, 100, 0.3), cand(500, 500, 0.8)])
    assert (s.x, s.y) == (500, 500) and not s.interpolated


def test_rejects_far_false_positive_and_keeps_track():
    bt = BallTracker(gate_px=100)
    for i in range(10):
        bt.step([cand(100 + 10 * i, 200)])
    # a confident detection far away (e.g. a head) plus the true ball near prediction
    s = bt.step([cand(900, 900, 0.9), cand(200, 200, 0.4)])
    assert abs(s.x - 200) < 15


def test_coasts_through_gap_then_resets():
    bt = BallTracker(max_missing=3)
    for i in range(10):
        bt.step([cand(100 + 10 * i, 200)])
    states = [bt.step([]) for _ in range(3)]
    assert all(s.interpolated for s in states)
    # predicted positions keep moving in the direction of travel
    assert states[-1].x > states[0].x > 190
    assert bt.step([]) is None   # 4th miss > max_missing -> lost


def test_low_confidence_ignored():
    bt = BallTracker(min_conf=0.2)
    assert bt.step([cand(10, 10, 0.1)]) is None


def test_smooths_noisy_track():
    rng = np.random.default_rng(0)
    bt = BallTracker()
    errs = []
    for i in range(60):
        tx, ty = 100 + 8 * i, 300
        s = bt.step([cand(tx + rng.normal(0, 6), ty + rng.normal(0, 6))])
        if i > 10:
            errs.append(np.hypot(s.x - tx, s.y - ty))
    assert np.mean(errs) < 6   # filtered error below raw measurement noise (~8.5)
