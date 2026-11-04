import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from physiosight.exercises.machines import FrameResult  # noqa: E402
from physiosight.live import mirror_pose  # noqa: E402
from physiosight.render.overlay import draw_hud, draw_pose, head_oval  # noqa: E402
from physiosight.testing import synth  # noqa: E402


def _pose():
    seq = synth.render(synth.curl_script(1), fps=10, size=(640, 480))
    return next(o.pose for o in seq.observations())


def test_head_is_covered_by_default():
    pose = _pose()
    oval = head_oval(pose.xy)
    assert oval is not None
    frame = np.full((480, 640, 3), 255, np.uint8)
    draw_pose(frame, pose)
    assert tuple(frame[oval.centre[1], oval.centre[0]]) == (80, 80, 80)
    shown = np.full((480, 640, 3), 255, np.uint8)
    draw_pose(shown, pose, cover_face=False)
    assert tuple(shown[oval.centre[1], oval.centre[0]]) == (255, 255, 255)


def test_hud_draws_without_error():
    frame = np.zeros((480, 640, 3), np.uint8)
    res = FrameResult(0.0, True, False, 3, "hint.common.progress", "metric.knee", 95.0)
    draw_hud(frame, res, "Squat", 10, "reps", fault_active=True, complete=True)
    draw_hud(frame, None, "Squat", 10, "reps")
    assert frame.any()


def test_mirror_pose_flips_x_only():
    pose = _pose()
    m = mirror_pose(pose, 640)
    np.testing.assert_allclose(m.xy[:, 0], 640 - pose.xy[:, 0])
    np.testing.assert_allclose(m.xy[:, 1], pose.xy[:, 1])
