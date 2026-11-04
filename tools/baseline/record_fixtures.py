"""Record the legacy engine's behaviour on the synthetic suite as a JSON fixture.

Run once with the legacy code present (it needs mediapipe/opencv importable):

    python -m tools.baseline.record_fixtures            # writes tests/fixtures/legacy_baseline.json

The fixture lets every later commit compare the new engine against the
legacy behaviour without the legacy code or MediaPipe installed.
"""
from __future__ import annotations

import json
import subprocess

from tools.baseline.cases import CASES
from tools.baseline.legacy_adapter import ROOT, replay

OUT = ROOT / "tests" / "fixtures" / "legacy_baseline.json"


def record() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "exercise_analysis.py",
                                 "legacy/exercise_analysis.py"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    cases = []
    for case in CASES:
        frames = replay(case.sequence(), case.exercise, case.target)
        progress_changes, last = [], None
        for i, f in enumerate(frames):
            if f.progress != last:
                progress_changes.append([i, round(f.progress, 9)])
                last = f.progress
        cases.append({
            "name": case.name, "expect": case.expect, "n_frames": len(frames),
            "final_progress": round(frames[-1].progress, 9),
            "progress_changes": progress_changes if case.exercise != "plank" else progress_changes[::10],
            "faults": [[i, f.fault] for i, f in enumerate(frames) if f.fault],
        })
    return {"generated_by": "tools/baseline/record_fixtures.py", "legacy_commit": commit, "cases": cases}


def main() -> int:
    data = record()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"recorded {len(data['cases'])} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
