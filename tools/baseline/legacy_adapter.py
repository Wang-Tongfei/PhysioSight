"""Replay cached landmarks through the *legacy* engine (exercise_analysis.py).

Used to (1) create baseline traces from real videos before the legacy code is
retired and (2) check that the new spec-driven engine reproduces legacy
behaviour where that is intended. The legacy modules live at the repository
root and are imported unchanged.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = ROOT / "legacy" if (ROOT / "legacy" / "exercise_analysis.py").is_file() else ROOT


def legacy_available() -> bool:
    return (LEGACY_DIR / "exercise_analysis.py").is_file()


def _import_path() -> None:
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))

from physiosight.perception.cache import LandmarkSequence  # noqa: E402


@dataclass(frozen=True)
class LegacyFrame:
    t: float
    analysed: bool
    progress: float
    fault: str | None
    form_ok: bool
    status: str
    metric_value: float | None


def replay(seq: LandmarkSequence, exercise: str, target: float = 10, arm: str = "right",
           stop_at_target: bool = True) -> list[LegacyFrame]:
    _import_path()
    from config import Config  # legacy
    from exercise_analysis import create_analyzer  # legacy
    from pose_estimator import PoseResult  # legacy

    cfg = Config()
    cfg.target_reps = int(target)
    cfg.plank_target_seconds = float(target)
    analyzer = create_analyzer(exercise, cfg, arm)
    out: list[LegacyFrame] = []
    w, h = seq.width, seq.height
    for i in range(len(seq)):
        t = float(seq.t[i])
        if stop_at_target and analyzer.is_complete:
            break  # the legacy loops stop analysing once the target is reached
        if not seq.present[i]:
            # the legacy loop does not call update() for frames without a pose
            out.append(LegacyFrame(t, False, float(analyzer.progress), None, True, "", None))
            continue
        pts = [(float(x) / w, float(y) / h) for x, y in seq.xy[i]]
        lm = PoseResult(pts, list(map(float, seq.visibility[i])), list(map(float, seq.presence[i])))
        r = analyzer.update(lm, now=t)
        out.append(LegacyFrame(t, True, float(analyzer.progress), r.fault, r.form_ok, r.status,
                               r.metric_value))
    return out
