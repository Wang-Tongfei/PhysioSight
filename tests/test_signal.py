import numpy as np

from physiosight.signal.filters import OneEuroFilter


def test_one_euro_reduces_jitter_on_static_signal():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 100 + rng.normal(0, 2, 300)
    out = np.array([f(np.array([x]), i / 30)[0] for i, x in enumerate(raw)])
    assert out[30:].std() < raw[30:].std() / 2


def test_one_euro_is_frame_rate_consistent():
    def run(fps):
        f = OneEuroFilter(1.0, 0.0)
        t = np.arange(0, 2, 1 / fps)
        out = [f(np.array([1.0 if ti > 0.5 else 0.0]), ti)[0] for ti in t]
        return out[-1]
    assert abs(run(10) - run(60)) < 0.02


def test_masked_samples_do_not_update_state():
    f = OneEuroFilter(1.0, 0.0)
    f(np.array([0.0]), 0.0)
    f(np.array([1000.0]), 0.033, mask=np.array([False]))
    assert abs(f(np.array([0.0]), 0.066)[0]) < 1e-9
