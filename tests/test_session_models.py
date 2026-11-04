from datetime import datetime

import pytest
from pydantic import ValidationError

from physiosight.session.models import (
    AngleExtrema,
    EngineInfo,
    FaultEvent,
    RepOutcome,
    RepRecord,
    Session,
    SessionSummary,
    SourceInfo,
    TrackingQuality,
)


def _session(**kw):
    rep = RepRecord(index=1, started_at_s=1.0, ended_at_s=2.5, outcome=RepOutcome.VALID, reached_target=True,
                    angles={"knee_flexion_left": AngleExtrema(min_deg=5, max_deg=95)})
    base = dict(
        exercise_id="squat", exercise_kind="repetition", source=SourceInfo(kind="video", width=1280, height=720),
        engine=EngineInfo(name="physiosight", version="0", pose_backend="test", spec_id="squat", spec_version="1"),
        tracking=TrackingQuality(frames_total=100, frames_with_pose=90, frames_reliable=80),
        summary=SessionSummary(target=10, achieved=1, attempts=1, completed=False, duration_s=3.0),
        reps=[rep],
    )
    base.update(kw)
    return Session(**base)


def test_valid_session_roundtrips():
    s = _session()
    assert Session.model_validate_json(s.model_dump_json()) == s
    assert isinstance(s.created_at, datetime) and s.reps[0].duration_s == pytest.approx(1.5)
    assert s.tracking.reliable_ratio == pytest.approx(0.8)


def test_valid_rep_cannot_have_faults():
    with pytest.raises(ValidationError):
        RepRecord(index=1, started_at_s=0, ended_at_s=1, outcome="valid", reached_target=True, faults=["x"])


def test_summary_must_match_reps():
    with pytest.raises(ValidationError):
        _session(summary=SessionSummary(target=10, achieved=2, attempts=1, completed=False, duration_s=3))


def test_fault_codes_are_identifiers():
    with pytest.raises(ValidationError):
        FaultEvent(t_s=0, code="Bad Code")


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        SourceInfo(kind="video", patient_name="x")


def test_tracking_counts_ordered():
    with pytest.raises(ValidationError):
        TrackingQuality(frames_total=10, frames_with_pose=11, frames_reliable=1)
