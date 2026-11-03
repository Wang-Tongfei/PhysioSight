"""Landmark sequences on disk (``.npz``).

Caching pose output decouples the analysis layer from the pose model: state
machine changes are compared on *identical* keypoints, regression tests run
in CI without MediaPipe, and research code can swap backends and diff
results frame by frame.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from physiosight.perception.base import SKELETONS, Observation, PoseFrame

FORMAT_VERSION = 1


@dataclass
class LandmarkSequence:
    skeleton: str
    width: int
    height: int
    t: np.ndarray  # (N,)
    present: np.ndarray  # (N,) bool
    xy: np.ndarray  # (N, K, 2) pixels, NaN where absent
    visibility: np.ndarray  # (N, K)
    presence: np.ndarray  # (N, K)
    world: np.ndarray | None = None  # (N, K, 3)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, k = len(self.t), len(SKELETONS[self.skeleton])
        if self.xy.shape != (n, k, 2):
            raise ValueError(f"xy must be ({n}, {k}, 2), got {self.xy.shape}")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("timestamps must be strictly increasing")

    def __len__(self) -> int:
        return len(self.t)

    @property
    def nominal_fps(self) -> float | None:
        if len(self.t) < 2:
            return None
        return float((len(self.t) - 1) / (self.t[-1] - self.t[0]))

    def observations(self) -> Iterator[Observation]:
        for i in range(len(self.t)):
            t = float(self.t[i])
            if not self.present[i]:
                yield Observation(t, None)
                continue
            yield Observation(
                t,
                PoseFrame(
                    t=t,
                    skeleton=self.skeleton,
                    xy=self.xy[i],
                    visibility=self.visibility[i],
                    presence=self.presence[i],
                    image_size=(self.width, self.height),
                    world=None if self.world is None else self.world[i],
                ),
            )

    def to_skeleton(self, skeleton: str) -> LandmarkSequence:
        """Keep only the keypoints that exist (by name) in another skeleton."""
        src, dst = SKELETONS[self.skeleton], SKELETONS[skeleton]
        missing = [n for n in dst if n not in src]
        if missing:
            raise ValueError(f"{self.skeleton} lacks keypoints {missing} needed by {skeleton}")
        idx = [src.index(n) for n in dst]
        return LandmarkSequence(
            skeleton, self.width, self.height, self.t, self.present, self.xy[:, idx],
            self.visibility[:, idx], self.presence[:, idx],
            None if self.world is None else self.world[:, idx], {**self.meta, "converted_from": self.skeleton},
        )

    @classmethod
    def from_frames(
        cls,
        frames: list[tuple[float, PoseFrame | None]],
        skeleton: str,
        width: int,
        height: int,
        meta: dict[str, Any] | None = None,
    ) -> LandmarkSequence:
        k = len(SKELETONS[skeleton])
        n = len(frames)
        xy = np.full((n, k, 2), np.nan)
        vis = np.zeros((n, k))
        pres = np.zeros((n, k))
        has_world = any(p is not None and p.world is not None for _, p in frames)
        world = np.full((n, k, 3), np.nan) if has_world else None
        present = np.zeros(n, dtype=bool)
        for i, (_, p) in enumerate(frames):
            if p is None:
                continue
            present[i] = True
            xy[i], vis[i], pres[i] = p.xy, p.visibility, p.presence
            if world is not None and p.world is not None:
                world[i] = p.world
        return cls(
            skeleton, width, height, np.array([t for t, _ in frames], float),
            present, xy, vis, pres, world, dict(meta or {}),
        )

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "t": self.t, "present": self.present, "xy": self.xy,
            "visibility": self.visibility, "presence": self.presence,
        }
        if self.world is not None:
            arrays["world"] = self.world
        header = {
            "format_version": FORMAT_VERSION, "skeleton": self.skeleton,
            "width": self.width, "height": self.height, "meta": self.meta,
        }
        arrays["header"] = np.array(json.dumps(header))
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> LandmarkSequence:
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            if header["format_version"] != FORMAT_VERSION:
                raise ValueError(f"unsupported landmark file version {header['format_version']}")
            return cls(
                header["skeleton"], header["width"], header["height"],
                data["t"], data["present"], data["xy"], data["visibility"],
                data["presence"], data["world"] if "world" in data else None,
                header.get("meta", {}),
            )
