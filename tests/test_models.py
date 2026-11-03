from pathlib import Path

import pytest

from physiosight.perception.models import HAND_MODEL, POSE_MODEL, model_path


def test_models_resolve_in_a_checkout():
    for name in (POSE_MODEL, HAND_MODEL):
        assert model_path(name).is_file()


def test_env_override(tmp_path, monkeypatch):
    (tmp_path / POSE_MODEL).write_bytes(b"x")
    monkeypatch.setenv("PHYSIOSIGHT_MODELS", str(tmp_path))
    assert model_path(POSE_MODEL) == Path(tmp_path) / POSE_MODEL


def test_missing_model_names_the_search_path():
    with pytest.raises(FileNotFoundError, match="PHYSIOSIGHT_MODELS"):
        model_path("nope.task")
