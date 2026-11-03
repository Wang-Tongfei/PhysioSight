import math

import pytest

from physiosight.exercises.expr import Expr, ExprError


@pytest.mark.parametrize("src,env,expected", [
    ("1 + 2 * 3", {}, 7.0),
    ("(1 + 2) * 3", {}, 9.0),
    ("-x + 1", {"x": 2.0}, -1.0),
    ("abs(a - b) > 20", {"a": 100.0, "b": 130.0}, 1.0),
    ("min(a, b) >= 160 and max(a, b) < 181", {"a": 170.0, "b": 165.0}, 1.0),
    ("not x or y", {"x": 1.0, "y": 0.0}, 0.0),
    ("mean(a, b, c)", {"a": 1.0, "b": 2.0, "c": 3.0}, 2.0),
    ("knee < p.depth", {"knee": 90.0, "p.depth": 100.0}, 1.0),
    ("true and not false", {}, 1.0),
])
def test_evaluates(src, env, expected):
    assert Expr.parse(src).evaluate(env) == pytest.approx(expected)


def test_nan_never_satisfies_a_condition():
    e = Expr.parse("knee < 100")
    assert not e.truth({"knee": math.nan})
    assert not Expr.parse("knee >= 100").truth({"knee": math.nan})
    assert Expr.parse("not (knee < 100)").truth({"knee": math.nan})


@pytest.mark.parametrize("bad", [
    "__import__('os')", "a.__class__", "1 < 2 < 3", "foo(1)", "abs(1, 2)", "a +", "(a", "a b",
    "lambda: 1", "a; b", "a[0]", "'x'",
])
def test_rejects_unsafe_or_malformed(bad):
    with pytest.raises(ExprError):
        Expr.parse(bad)


def test_names():
    assert Expr.parse("abs(knee_l - knee_r) > p.max").names() == {"knee_l", "knee_r", "p.max"}
