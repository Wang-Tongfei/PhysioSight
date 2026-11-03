import json

import pytest
from pydantic import ValidationError

from physiosight.pipeline import analyze_sequence
from physiosight.session.models import Session
from physiosight.testing import synth


def test_session_roundtrip_and_schema(tmp_path):
    s = analyze_sequence(synth.render(synth.squat_script(2)), "squat", 10).session
    again = Session.model_validate_json(s.model_dump_json())
    assert again == s
    schema = Session.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "0.1.0"
    json.dumps(schema)


def test_session_is_internally_consistent():
    s = analyze_sequence(synth.render(synth.squat_script(2)), "squat", 10).session
    data = json.loads(s.model_dump_json())
    data["summary"]["achieved"] = 5
    with pytest.raises(ValidationError):
        Session.model_validate(data)


def test_session_contains_no_identity_or_images():
    s = analyze_sequence(synth.render(synth.squat_script(1)), "squat", 10).session
    keys = set(json.loads(s.model_dump_json()))
    assert not keys & {"patient_id", "name", "frames", "landmarks", "video"}
