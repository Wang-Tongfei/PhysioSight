"""Backend-independent pose representation.

Every pose backend (MediaPipe, RTMPose via ONNX Runtime, a cached file, a
synthetic generator) produces :class:`PoseFrame` objects. Coordinates are in
*pixels* so that angles are not distorted by non-square frames; the old engine
computed angles on normalised (x, y), which skews angles whenever
width != height.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

BLAZEPOSE_33: tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)

COCO_17: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle",
)

SKELETONS: dict[str, tuple[str, ...]] = {"blazepose33": BLAZEPOSE_33, "coco17": COCO_17}


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """One frame of keypoints for a single person.

    ``xy``: (K, 2) pixel coordinates, image origin top-left, y down.
    ``visibility`` / ``presence``: (K,) in [0, 1]; backends without one of
    these fill it with ones.
    ``world``: optional (K, 3) metric coordinates (MediaPipe world landmarks,
    or a 3D lifting model), y down, hip-centred.
    """

    t: float
    skeleton: str
    xy: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    world: np.ndarray | None = None

    def index(self, name: str) -> int:
        return SKELETONS[self.skeleton].index(name)

    def reliable(
        self, indices: Iterable[int], min_visibility: float, min_presence: float
    ) -> bool:
        idx = np.fromiter(indices, dtype=int)
        if idx.size == 0:
            return True
        w, h = self.image_size
        pts = self.xy[idx]
        finite = np.all(np.isfinite(pts))
        return bool(
            finite
            and np.all(self.visibility[idx] >= min_visibility)
            and np.all(self.presence[idx] >= min_presence)
        ) and w > 0 and h > 0


@dataclass(frozen=True, slots=True)
class Observation:
    """What the analysis layer sees per frame: a timestamp and maybe a pose."""

    t: float
    pose: PoseFrame | None


class PoseBackend(Protocol):
    """Uniform interface for pose models.

    ``process`` must be called with strictly increasing timestamps (seconds).
    """

    name: str
    version: str | None
    skeleton: str

    def process(self, frame_bgr: np.ndarray, t: float) -> PoseFrame | None: ...

    def close(self) -> None: ...


class FrameSource(Protocol):
    width: int
    height: int
    nominal_fps: float | None
    kind: str

    def frames(self) -> Iterator[tuple[float, np.ndarray]]: ...

    def close(self) -> None: ...
