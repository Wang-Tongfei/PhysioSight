import pytest
import yaml

from physiosight.exercises.spec import list_builtin_specs, load_spec, resolve_params


def test_builtin_specs_load():
    assert set(list_builtin_specs()) >= {"bicep_curl", "squat", "pushup", "plank"}
    for name in list_builtin_specs():
        spec, sha = load_spec(name)
        assert spec.id == name and len(sha) == 64


def test_side_substitution():
    spec, _ = load_spec("bicep_curl", side="left")
    assert spec.required[0] == "left_shoulder"
    assert "elbow_flexion_left" in spec.record
    with pytest.raises(ValueError):
        load_spec("bicep_curl", side="bilateral")


def test_threshold_layers_and_bounds():
    spec, _ = load_spec("squat")
    p = resolve_params(spec, {"depth_angle": 110}, {"max_asymmetry": 15})
    assert p["depth_angle"] == 110 and p["max_asymmetry"] == 15 and p["stand_angle"] == 160
    with pytest.raises(KeyError):
        resolve_params(spec, {"depht_angle": 110})
    with pytest.raises(ValueError):
        resolve_params(spec, {"depth_angle": 5})


def _write(tmp_path, mutate):
    spec_path = tmp_path / "x.yaml"
    raw = yaml.safe_load(open(load_spec.__globals__["_builtin_dir"]() / "squat.yaml"))
    mutate(raw)
    spec_path.write_text(yaml.safe_dump(raw))
    return spec_path


def test_unknown_names_rejected_at_load(tmp_path):
    def bad(raw):
        raw["target"]["when"] = "knee_x < p.depth_angle"
    with pytest.raises(ValueError, match="unknown names"):
        load_spec(_write(tmp_path, bad))


def test_unknown_keypoint_rejected(tmp_path):
    from physiosight.exercises.machines import create_analyzer

    def bad(raw):
        raw["features"]["knee_l"]["points"][0] = "left_hipp"
    spec, _ = load_spec(_write(tmp_path, bad))
    with pytest.raises(ValueError, match="not in skeleton"):
        create_analyzer(spec, 10)
