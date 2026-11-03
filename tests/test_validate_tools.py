"""The local validation script runs end to end on cached landmarks (no video needed)."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

from physiosight.pipeline import analyze_sequence
from physiosight.testing import synth

ROOT = Path(__file__).resolve().parents[1]


def test_counts_report_from_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for i, (sc, true) in enumerate([(synth.squat_script(3), 3), (synth.squat_script(2, depth=120), 0)]):
        synth.render(sc).save(cache / f"v{i}.npz")
        rows.append({"video": f"v{i}.mp4", "exercise": "squat", "side": "", "true_reps": true,
                     "true_faults": "" if true else "shallow_squat"})
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "r.md"
    subprocess.run([sys.executable, str(ROOT / "tools/validate/evaluate.py"), "counts", str(labels),
                    "--cache", str(cache), "-o", str(out)], check=True, capture_output=True)
    text = out.read_text()
    assert "exact: 1.00" in text and "recall 1.00" in text


def test_angles_report(tmp_path):
    pytest.importorskip("scipy")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    ref_rows = []
    for i, depth in enumerate([70, 80, 90, 95]):
        s = analyze_sequence(synth.render(synth.squat_script(1, depth=depth)), "squat", 10).session
        (sessions / f"s{i}.json").write_text(s.model_dump_json())
        ref_rows.append({"session_file": f"s{i}.json", "rep_index": 1, "angle": "knee_flexion_left",
                         "reference_deg": 180 - depth + 2.0})
    ref = tmp_path / "ref.csv"
    with open(ref, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ref_rows[0]))
        w.writeheader()
        w.writerows(ref_rows)
    out = tmp_path / "a.md"
    subprocess.run([sys.executable, str(ROOT / "tools/validate/evaluate.py"), "angles", str(ref),
                    "--sessions", str(sessions), "-o", str(out)], check=True, capture_output=True)
    assert "| knee_flexion_left | 4 |" in out.read_text() and "-2.0" in out.read_text()
