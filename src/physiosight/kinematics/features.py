"""Geometric features computed from a :class:`PoseFrame`.

Feature *types* are implemented here; *which* features an exercise uses is
declared in its YAML spec. A point reference is either a keypoint name or a
list of names (their midpoint).

All image-space computations use pixel coordinates, so angles are correct
for any aspect ratio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from physiosight.perception.base import SKELETONS, PoseFrame

PointRef = Union[str, list[str]]


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` between rays b->a and b->c, degrees in [0, 180].

    Works for 2D and 3D points. Uses atan2(|u x v|, u . v), which is
    numerically stable near 0 and 180 degrees (unlike acos).
    """
    u, v = np.asarray(a, float) - b, np.asarray(c, float) - b
    if u.shape[-1] == 2:
        cross = abs(u[0] * v[1] - u[1] * v[0])
    else:
        cross = float(np.linalg.norm(np.cross(u, v)))
    dot = float(np.dot(u, v))
    if cross == 0.0 and dot == 0.0:
        return float("nan")  # degenerate: coincident points
    return math.degrees(math.atan2(cross, dot))


def incline_from_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle of segment a-b from the image horizontal, [0, 90]."""
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    ang = abs(math.degrees(math.atan2(dy, dx)))
    return min(ang, 180.0 - ang)


class _Feature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JointAngle(_Feature):
    type: Literal["joint_angle"]
    points: tuple[PointRef, PointRef, PointRef]
    space: Literal["image", "world"] = "image"


class SegmentIncline(_Feature):
    type: Literal["segment_incline"]
    points: tuple[PointRef, PointRef]


class VerticalOffset(_Feature):
    """(y_a - y_b) divided by a scale; positive when ``a`` is below ``b``.

    ``scale: image_height`` reproduces the legacy behaviour; ``segment``
    divides by the length of ``scale_points`` (body-size invariant).
    """

    type: Literal["vertical_offset"]
    points: tuple[PointRef, PointRef]
    scale: Literal["image_height", "segment"] = "image_height"
    scale_points: tuple[PointRef, PointRef] | None = None


class DistanceRatio(_Feature):
    """|a-b| / |c-d|, e.g. shoulder width over torso length for view checks."""

    type: Literal["distance_ratio"]
    points: tuple[PointRef, PointRef, PointRef, PointRef]


FeatureDef = Annotated[
    JointAngle | SegmentIncline | VerticalOffset | DistanceRatio,
    Field(discriminator="type"),
]


def _names(ref: PointRef) -> list[str]:
    return [ref] if isinstance(ref, str) else list(ref)


@dataclass(frozen=True)
class CompiledFeatures:
    """Feature definitions bound to a skeleton's keypoint indices."""

    skeleton: str
    defs: dict[str, FeatureDef]
    _idx: dict[str, list[tuple[int, ...]]]

    @classmethod
    def compile(cls, defs: dict[str, FeatureDef], skeleton: str) -> CompiledFeatures:
        names = SKELETONS[skeleton]
        idx: dict[str, list[tuple[int, ...]]] = {}
        for fname, fdef in defs.items():
            refs = list(fdef.points)
            if isinstance(fdef, VerticalOffset) and fdef.scale == "segment":
                if fdef.scale_points is None:
                    raise ValueError(f"feature {fname}: scale 'segment' needs scale_points")
                refs += list(fdef.scale_points)
            resolved = []
            for ref in refs:
                missing = [n for n in _names(ref) if n not in names]
                if missing:
                    raise ValueError(
                        f"feature {fname}: keypoints {missing} not in skeleton {skeleton}"
                    )
                resolved.append(tuple(names.index(n) for n in _names(ref)))
            idx[fname] = resolved
        return cls(skeleton, dict(defs), idx)

    def keypoints(self) -> set[int]:
        return {i for refs in self._idx.values() for ref in refs for i in ref}

    def compute(self, pose: PoseFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for fname, fdef in self.defs.items():
            refs = self._idx[fname]
            use_world = isinstance(fdef, JointAngle) and fdef.space == "world"
            if use_world and pose.world is None:
                out[fname] = float("nan")
                continue
            coords = pose.world if use_world else pose.xy
            assert coords is not None
            pts = [coords[list(r)].mean(axis=0) for r in refs]
            out[fname] = _evaluate(fdef, pts, pose)
        return out


def _evaluate(fdef: FeatureDef, pts: list[np.ndarray], pose: PoseFrame) -> float:
    if isinstance(fdef, JointAngle):
        return interior_angle(*pts[:3])
    if isinstance(fdef, SegmentIncline):
        return incline_from_horizontal(pts[0], pts[1])
    if isinstance(fdef, VerticalOffset):
        dy = float(pts[0][1] - pts[1][1])
        if fdef.scale == "image_height":
            return dy / pose.image_size[1]
        seg = float(np.linalg.norm(pts[2] - pts[3]))
        return dy / seg if seg > 0 else float("nan")
    if isinstance(fdef, DistanceRatio):
        den = float(np.linalg.norm(pts[2] - pts[3]))
        return float(np.linalg.norm(pts[0] - pts[1])) / den if den > 0 else float("nan")
    raise TypeError(fdef)
