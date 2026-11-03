"""Property tests (Hypothesis) for invariants the analysis must keep."""
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from physiosight.pipeline import analyze_sequence
from physiosight.testing import synth

FAST = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def count(script, fps, **kw):
    seq = synth.render(script, fps=fps, **kw)
    return analyze_sequence(seq, script.exercise, 50, stop_at_target=False).session.summary.achieved


@FAST
@given(reps=st.integers(1, 4), depth=st.floats(60, 95), down=st.floats(0.5, 1.5), up=st.floats(0.5, 1.5),
       pause=st.floats(0.3, 1.0))
def test_squat_count_is_frame_rate_invariant(reps, depth, down, up, pause):
    sc = synth.squat_script(reps, depth=depth, down_s=down, up_s=up, pause_s=pause)
    counts = {fps: count(sc, fps) for fps in (10, 30, 60)}
    assert len(set(counts.values())) == 1, counts
    assert counts[30] == reps


@FAST
@given(reps=st.integers(1, 4), depth=st.floats(105, 150), fps=st.sampled_from([10, 30, 60]))
def test_never_counts_without_reaching_depth(reps, depth, fps):
    assert count(synth.squat_script(reps, depth=depth), fps) == 0


@FAST
@given(reps=st.integers(1, 3), fps=st.sampled_from([10, 30, 60]), exercise=st.sampled_from(["squat", "bicep_curl", "pushup"]))
def test_low_confidence_never_advances(reps, fps, exercise):
    sc = {"squat": synth.squat_script, "bicep_curl": synth.curl_script, "pushup": synth.pushup_script}[exercise](reps)
    seq = synth.render(sc, fps=fps, dropout=lambda t: True)
    s = analyze_sequence(seq, exercise, 50).session
    assert s.summary.achieved == 0 and s.tracking.frames_reliable == 0


@FAST
@given(w=st.integers(320, 1920), h=st.integers(320, 1920))
def test_count_does_not_depend_on_aspect_ratio(w, h):
    sc = synth.squat_script(2, depth=92)
    assert count(sc, 30, size=(w, h)) == 2


@FAST
@given(seed=st.integers(0, 10_000))
def test_counting_is_robust_to_keypoint_noise_with_smoothing(seed):
    from physiosight.signal.filters import LandmarkSmoother

    sc = synth.squat_script(3, depth=85)
    seq = synth.render(sc, fps=30, noise_px=3.0, seed=seed)
    s = analyze_sequence(seq, "squat", 50, stop_at_target=False, smoother=LandmarkSmoother()).session
    assert s.summary.achieved == 3
