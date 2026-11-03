import json
from pathlib import Path

from physiosight.cli import main
from physiosight.session.models import Session
from physiosight.testing import synth

SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "session.schema.json"


def test_committed_json_schema_matches_model():
    """schema/session.schema.json is the contract consumers generate types from."""
    assert json.loads(SCHEMA.read_text()) == Session.model_json_schema(), \
        "run: physiosight schema -o schema/session.schema.json"


def test_specs_lists_builtins(capsys):
    assert main(["specs"]) == 0
    out = capsys.readouterr().out
    for name in ("bicep_curl", "squat", "plank", "pushup", "shoulder_flexion"):
        assert name in out


def test_analyze_writes_a_valid_session(tmp_path, capsys):
    seq = synth.render(synth.squat_script(3), fps=30)
    seq.save(tmp_path / "s.npz")
    out = tmp_path / "s.json"
    assert main(["analyze", str(tmp_path / "s.npz"), "-e", "squat", "-o", str(out), "--lang", "zh"]) == 0
    s = Session.model_validate_json(out.read_text())
    assert s.summary.achieved == 3 and s.summary.target == 10
    assert "深蹲" in capsys.readouterr().err


def test_analyze_threshold_overrides(tmp_path):
    seq = synth.render(synth.squat_script(2, depth=110), fps=30)
    seq.save(tmp_path / "s.npz")
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"depth_angle": 115}))  # patient calibration layer
    out = tmp_path / "s.json"
    main(["analyze", str(tmp_path / "s.npz"), "-e", "squat", "--params", str(calib), "-o", str(out)])
    s = Session.model_validate_json(out.read_text())
    assert s.summary.achieved == 2 and s.engine.params["depth_angle"] == 115
