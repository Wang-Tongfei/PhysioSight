import numpy as np
import pytest

pytest.importorskip("scipy")

from physiosight.eval.agreement import bland_altman, icc, interpret_icc
from physiosight.eval.counting import mae_normalized, obo
from physiosight.eval.detection import per_rep_prf

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])


@pytest.mark.parametrize("form,expected", [("1,1", 0.17), ("2,1", 0.29), ("3,1", 0.71)])
def test_icc_matches_shrout_fleiss(form, expected):
    r = icc(SF, form)
    assert r.value == pytest.approx(expected, abs=0.005)
    assert r.ci95[0] < r.value < r.ci95[1]


def test_icc_perfect_agreement():
    x = np.arange(10.0)
    assert icc(np.c_[x, x + 1e-9 * np.random.default_rng(0).normal(size=10)], "2,1").value > 0.999
    assert interpret_icc(0.95) == "excellent" and interpret_icc(0.3) == "poor"


def test_bland_altman():
    ref = np.linspace(0, 100, 50)
    ba = bland_altman(ref + 2.0, ref)
    assert ba.bias == pytest.approx(2.0) and ba.sd == pytest.approx(0, abs=1e-9)
    prop = bland_altman(ref * 1.1, ref)
    assert prop.proportional_p < 1e-6


def test_counting():
    assert mae_normalized([10, 8], [10, 10]) == pytest.approx(0.1)
    assert obo([10, 8, 11], [10, 10, 10]) == pytest.approx(2 / 3)


def test_detection():
    out = per_rep_prf([{"a"}, {"a", "b"}, set()], [{"a"}, {"b"}, {"a"}])
    assert out["a"].precision == pytest.approx(0.5) and out["a"].recall == pytest.approx(0.5)
    assert out["b"].f1 == pytest.approx(1.0)
