"""Locate the pose/hand model files.

Search order: ``$PHYSIOSIGHT_MODELS`` (a directory), the copy bundled inside
the installed wheel (``physiosight/models/``), then ``models/`` of a source
checkout. The MediaPipe ``.task`` models are Apache-2.0 licensed.
"""
from __future__ import annotations

import os
from pathlib import Path

POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
_PACKAGE = Path(__file__).resolve().parents[1]


def candidates(name: str) -> list[Path]:
    out = []
    env = os.environ.get("PHYSIOSIGHT_MODELS")
    if env:
        out.append(Path(env) / name)
    out.append(_PACKAGE / "models" / name)  # wheel install (force-included)
    out.append(_PACKAGE.parents[1] / "models" / name)  # source checkout: <repo>/src/physiosight -> <repo>/models
    return out


def model_path(name: str = POSE_MODEL) -> Path:
    for path in candidates(name):
        if path.is_file():
            return path
    searched = ", ".join(str(p) for p in candidates(name))
    raise FileNotFoundError(f"model {name!r} not found (searched {searched}); set PHYSIOSIGHT_MODELS")
