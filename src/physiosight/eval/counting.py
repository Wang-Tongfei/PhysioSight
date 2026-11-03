"""Repetition-counting metrics as used on RepCount (TransRAC, CVPR 2022; PoseRAC, 2023).

* ``mae_normalized`` = mean(|pred - gt| / gt)  (the "MAE" reported on RepCount)
* ``obo``            = fraction of videos with |pred - gt| <= 1 (off-by-one accuracy)
Report both: MAE rewards being close on long videos, OBO rewards being
exactly right on short ones. For rehabilitation sets (5-15 reps) OBO and the
absolute error are the clinically meaningful numbers.
"""
from __future__ import annotations

import numpy as np


def mae_normalized(pred, gt) -> float:
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    if np.any(g <= 0):
        raise ValueError("ground-truth counts must be > 0 for normalised MAE")
    return float(np.mean(np.abs(p - g) / g))


def mae_absolute(pred, gt) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(gt, float))))


def obo(pred, gt) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(gt, float)) <= 1))


def exact(pred, gt) -> float:
    return float(np.mean(np.asarray(pred, float) == np.asarray(gt, float)))
