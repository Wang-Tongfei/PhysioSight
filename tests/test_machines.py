"""State machines on synthetic landmark sequences with known ground truth."""
import pytest

from physiosight.events import FaultDetected
from physiosight.pipeline import analyze_sequence
from physiosight.session.models import RepOutcome
from physiosight.testing import synth


def run(script, fps=30.0, **kw):
    seq = synth.render(script, fps=fps, **{k: v for k, v in kw.items() if k in ("noise_px", "seed", "dropout", "size")})
    return analyze_sequence(seq, script.exercise, kw.get("target", 10), stop_at_target=False, keep_frames=True)


@pytest.mark.parametrize("script", [synth.squat_script(3), synth.curl_script(3), synth.pushup_script(3)],
                         ids=lambda s: s.exercise)
def test_counts_clean_repetitions(script):
    s = run(script).session
    assert s.summary.achieved == 3 and s.summary.attempts == 3
    assert all(r.outcome is RepOutcome.VALID for r in s.reps)
    assert not s.faults


@pytest.mark.parametrize("script,fault", [
    (synth.squat_script(2, depth=120), "shallow_squat"),
    (synth.squat_script(2, asym=30), "uneven_squat"),
    (synth.squat_script(2, down_s=0.2, up_s=0.2), "movement_too_fast"),
    (synth.curl_script(2, top=70), "partial_curl"),
    (synth.curl_script(2, drift=50), "upper_arm_drift"),
    (synth.pushup_script(2, bottom=120), "shallow_pushup"),
])
def test_faulty_repetitions_are_not_counted(script, fault):
    s = run(script).session
    assert s.summary.achieved == 0
    assert fault in {f.code for f in s.faults}
    assert all(fault in r.faults for r in s.reps)


def test_rep_records_angles_and_timing():
    s = run(synth.squat_script(1, depth=90)).session
    rep = s.reps[0]
    knee = rep.angles["knee_flexion_left"]
    assert knee.max_deg == pytest.approx(90, abs=1.0)  # 180 - 90 interior
    assert 1.0 < rep.duration_s < 3.0


def test_tracking_loss_aborts_the_current_repetition():
    script = synth.squat_script(1)
    # occlude for 0.3 s in the middle of the descent
    s = run(script, dropout=lambda t: 1.4 < t < 1.7).session
    assert s.summary.achieved == 0
    assert s.reps[0].outcome is RepOutcome.ABORTED
    assert s.tracking.tracking_lost_events == 1


def test_plank_accumulates_only_good_form():
    good = analyze_sequence(synth.render(synth.plank_script(6)), "plank", 30).session
    assert good.summary.achieved == pytest.approx(6, abs=0.05)
    sag = analyze_sequence(synth.render(synth.plank_script(6, sag=60)), "plank", 30).session
    assert sag.summary.achieved == 0
    assert [f.code for f in sag.faults] == ["plank_body_alignment"]


def test_stop_at_target():
    seq = synth.render(synth.squat_script(5))
    s = analyze_sequence(seq, "squat", 2).session
    assert s.summary.achieved == 2 and s.summary.completed


def test_fault_events_are_edge_triggered():
    r = run(synth.curl_script(1, drift=50))
    drift = [e for f in r.frames for e in f.events if isinstance(e, FaultDetected) and e.code == "upper_arm_drift"]
    assert 1 <= len(drift) <= 2  # once when it starts (+ once as the end-of-rep reason)


def test_view_check_flags_wrong_view_without_blocking_counting():
    body = synth.Body(scale=327, depth_offset=90.0)  # wide shoulders = facing the camera
    seq = synth.render(synth.shoulder_flexion_script(2), fps=30, body=body)
    s = analyze_sequence(seq, "shoulder_flexion", 10, stop_at_target=False).session
    assert s.summary.achieved == 2
    assert s.tracking.frames_view_ok == 0 and all(r.view_ok is False for r in s.reps)
    side = analyze_sequence(synth.render(synth.shoulder_flexion_script(2)), "shoulder_flexion", 10).session
    assert all(r.view_ok for r in side.reps)


def test_new_exercise_is_pure_yaml_shoulder_flexion():
    s = analyze_sequence(synth.render(synth.shoulder_flexion_script(3, top=160)), "shoulder_flexion", 10).session
    assert s.summary.achieved == 3
    peak = s.reps[0].angles["shoulder_flexion_right"].max_deg
    assert abs(peak - 160) < 1.0
    low = analyze_sequence(synth.render(synth.shoulder_flexion_script(2, top=120)), "shoulder_flexion", 10,
                           stop_at_target=False).session
    assert low.summary.achieved == 0 and {"insufficient_range"} <= {f.code for f in low.faults}
    bent = analyze_sequence(synth.render(synth.shoulder_flexion_script(2, elbow=120)), "shoulder_flexion", 10,
                            stop_at_target=False).session
    assert bent.summary.achieved == 0 and "elbow_bent" in {f.code for f in bent.faults}


@pytest.mark.parametrize("script", [synth.squat_script(3), synth.curl_script(2), synth.pushup_script(2)],
                         ids=lambda s: s.exercise)
def test_same_spec_runs_on_coco17_skeleton(script):
    """Specs reference keypoints by name, so a COCO-17 backend (e.g. RTMPose) works unchanged."""
    seq = synth.render(script, fps=30)
    coco = seq.to_skeleton("coco17")
    a = analyze_sequence(seq, script.exercise, 10).session
    b = analyze_sequence(coco, script.exercise, 10).session
    assert a.summary.achieved == b.summary.achieved == script.expected_valid


def test_frames_without_a_person_abort_the_repetition():
    """BC-3: legacy skipped pose-less frames, so a rep survived any time out of view."""
    seq = synth.render(synth.squat_script(1), fps=30)
    gone = (seq.t > 1.4) & (seq.t < 2.4)  # person leaves the frame mid-descent for 1 s
    seq.present[gone] = False
    s = analyze_sequence(seq, "squat", 10).session
    assert s.summary.achieved == 0 and s.reps[0].outcome is RepOutcome.ABORTED
    assert s.tracking.frames_with_pose == int((~gone).sum())
