"""Landmark signal conditioning.

* :class:`OneEuroFilter` - speed-adaptive low-pass filter (Casiez, Roussel &
  Vogel, CHI 2012). Low cutoff at low speed removes jitter, the cutoff rises
  with speed to limit lag. Time-stamped, so it behaves the same at any fps.
* :class:`LandmarkSmoother` - applies a One Euro filter per keypoint
  coordinate, skipping low-confidence keypoints (they keep their raw value
  and do not pollute the filter state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from physiosight.perception.base import PoseFrame


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorised 1-euro filter over an array of any shape."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be > 0")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float, mask: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        assert self._dx is not None
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self._x
        if mask is not None:
            # keep raw value + old state where the sample is not trusted
            x_hat = np.where(mask, x_hat, x)
            self._x = np.where(mask, x_hat, self._x)
            self._dx = np.where(mask, dx_hat, self._dx)
        else:
            self._x, self._dx = x_hat, dx_hat
        self._t = t
        return x_hat


@dataclass
class LandmarkSmoother:
    min_cutoff: float = 1.0
    beta: float = 0.01
    d_cutoff: float = 1.0
    min_visibility: float = 0.5
    reset_after_s: float = 0.5

    def __post_init__(self) -> None:
        self._filter = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
        self._last_t: float | None = None

    @property
    def description(self) -> str:
        return f"one_euro({self.min_cutoff:g},{self.beta:g},{self.d_cutoff:g})"

    def __call__(self, pose: PoseFrame | None, t: float) -> PoseFrame | None:
        if pose is None:
            return None
        if self._last_t is not None and t - self._last_t > self.reset_after_s:
            self._filter.reset()
        self._last_t = t
        mask = (pose.visibility >= self.min_visibility)[:, None] & np.isfinite(pose.xy)
        xy = self._filter(np.nan_to_num(pose.xy), t, mask)
        xy = np.where(np.isfinite(pose.xy), xy, np.nan)
        return replace(pose, xy=xy)
