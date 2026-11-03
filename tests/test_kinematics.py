import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiosight.kinematics.features import incline_from_horizontal, interior_angle


@given(st.floats(0.5, 179.5), st.floats(0, 360), st.floats(1, 500), st.floats(1, 500))
def test_interior_angle_recovers_constructed_angle(angle, rot, r1, r2):
    b = np.array([10.0, 20.0])
    a = b + r1 * np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
    c = b + r2 * np.array([math.cos(math.radians(rot + angle)), math.sin(math.radians(rot + angle))])
    assert interior_angle(a, b, c) == pytest.approx(angle, abs=1e-6)


def test_interior_angle_3d_and_degenerate():
    assert interior_angle(np.array([1, 0, 0.0]), np.zeros(3), np.array([0, 0, 1.0])) == pytest.approx(90)
    assert math.isnan(interior_angle(np.zeros(2), np.zeros(2), np.ones(2)))


@given(st.floats(1, 4000), st.floats(1, 4000), st.floats(0.5, 179.5))
def test_pixel_angles_do_not_depend_on_aspect_ratio(w, h, angle):
    """Legacy bug: angles on normalised (x/w, y/h) change with aspect ratio."""
    b = np.array([w / 2, h / 2])
    a = b + np.array([100.0, 0.0])
    c = b + 100 * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    assert interior_angle(a, b, c) == pytest.approx(angle, abs=1e-6)


def test_incline():
    assert incline_from_horizontal(np.array([0, 0.0]), np.array([10, 0.0])) == pytest.approx(0)
    assert incline_from_horizontal(np.array([0, 0.0]), np.array([-10, 10.0])) == pytest.approx(45)
    assert incline_from_horizontal(np.array([0, 0.0]), np.array([0, 10.0])) == pytest.approx(90)
