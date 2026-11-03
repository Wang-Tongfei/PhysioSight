"""New engine vs the recorded legacy baseline (tests/fixtures/legacy_baseline.json).

The fixture was recorded from the unmodified legacy engine by
tools/baseline/record_fixtures.py before the refactor. These tests need
neither the legacy code nor MediaPipe.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from tools.baseline.cases import CASES

from physiosight.events import FaultDetected
from physiosight.pipeline import analyze_sequence

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_baseline.json").read_text())
LEGACY = {c["name"]: c for c in FIXTURE["cases"]}


def _run(case):
    return analyze_sequence(case.sequence(), case.exercise, case.target, keep_frames=True).frames


def test_fixture_covers_every_case():
    assert set(LEGACY) == {c.name for c in CASES}


@pytest.mark.parametrize("case", [c for c in CASES if c.expect == "identical"], ids=lambda c: c.name)
def test_identical_to_legacy_frame_by_frame(case):
    legacy, frames = LEGACY[case.name], _run(case)
    assert len(frames) == legacy["n_frames"]
    faults = [[i, e.code] for i, f in enumerate(frames) for e in f.events if isinstance(e, FaultDetected)]
    assert faults == legacy["faults"]
    if case.exercise == "plank":
        for i, p in legacy["progress_changes"]:
            assert frames[i].progress == pytest.approx(p, abs=1e-9)
        assert frames[-1].progress == pytest.approx(legacy["final_progress"], abs=1e-9)
    else:
        changes, last = [], None
        for i, f in enumerate(frames):
            if f.progress != last:
                changes.append([i, f.progress])
                last = f.progress
        assert changes == legacy["progress_changes"]


@pytest.mark.parametrize("case", [c for c in CASES if c.expect != "identical"], ids=lambda c: c.name)
def test_intentional_change_fixes_legacy(case):
    """Documented in docs/BEHAVIOR_CHANGES.md under the case's BC id."""
    truth = case.script().expected_valid
    new = _run(case)[-1].progress
    assert new == truth
    assert LEGACY[case.name]["final_progress"] != truth


def test_legacy_reproduces_fixture():
    """Guards the fixture itself: replaying the legacy code gives the same result."""
    pytest.importorskip("mediapipe")
    from tools.baseline.legacy_adapter import legacy_available, replay

    if not legacy_available():
        pytest.skip("legacy engine not present")
    for case in CASES[:6] + [c for c in CASES if c.expect != "identical"]:
        frames = replay(case.sequence(), case.exercise, case.target)
        assert [[i, f.fault] for i, f in enumerate(frames) if f.fault] == LEGACY[case.name]["faults"]
        assert np.isclose(frames[-1].progress, LEGACY[case.name]["final_progress"])
