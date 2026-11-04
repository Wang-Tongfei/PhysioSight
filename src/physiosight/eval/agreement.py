"""Agreement between a measurement system and a reference (e.g. markerless vs optical mocap).

* ICC forms follow Shrout & Fleiss (1979) / McGraw & Wong (1996). For method
  comparison report ICC(2,1) (two-way random, absolute agreement, single
  measurement) and interpret it with Koo & Li (2016): <0.5 poor,
  0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent - always with its 95% CI.
* Bland-Altman (1986): bias and 95% limits of agreement, plus a test for
  proportional bias (slope of difference vs mean).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICCResult:
    form: str
    value: float
    ci95: tuple[float, float]
    f: float
    df1: int
    df2: int


def icc(ratings: np.ndarray, form: str = "2,1") -> ICCResult:
    """``ratings``: (n_subjects_or_trials, k_raters_or_methods)."""
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("ratings must be (n>=2, k>=2)")
    if np.isnan(x).any():
        raise ValueError("ratings contain NaN; drop incomplete rows first")
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc = ss_rows / (n - 1), ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_err) / (n * (k - 1))
    from scipy import stats

    alpha = 0.05
    if form == "1,1":
        value = (msr - msw) / (msr + (k - 1) * msw)
        f, df1, df2 = msr / msw, n - 1, n * (k - 1)
        fl, fu = f / stats.f.ppf(1 - alpha / 2, df1, df2), f * stats.f.ppf(1 - alpha / 2, df2, df1)
        ci = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    elif form == "3,1":
        value = (msr - mse) / (msr + (k - 1) * mse)
        f, df1, df2 = msr / mse, n - 1, (n - 1) * (k - 1)
        fl, fu = f / stats.f.ppf(1 - alpha / 2, df1, df2), f * stats.f.ppf(1 - alpha / 2, df2, df1)
        ci = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    elif form == "2,1":
        value = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
        f, df1, df2 = msr / mse, n - 1, (n - 1) * (k - 1)
        # McGraw & Wong (1996) approximate CI for absolute agreement
        a = k * value / (n * (1 - value))
        b = 1 + k * value * (n - 1) / (n * (1 - value))
        v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
        fs_u = stats.f.ppf(1 - alpha / 2, n - 1, v)
        fs_l = stats.f.ppf(1 - alpha / 2, v, n - 1)
        lo = n * (msr - fs_u * mse) / (fs_u * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (fs_l * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fs_l * msr)
        ci = (lo, hi)
    else:
        raise ValueError("form must be '1,1', '2,1' or '3,1'")
    return ICCResult(form, float(value), (float(ci[0]), float(ci[1])), float(f), df1, int(df2))


def interpret_icc(value: float) -> str:
    """Koo & Li (2016) bands."""
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"


@dataclass(frozen=True)
class BlandAltman:
    bias: float
    sd: float
    loa: tuple[float, float]
    proportional_slope: float
    proportional_p: float
    n: int


def bland_altman(system: np.ndarray, reference: np.ndarray) -> BlandAltman:
    from scipy import stats

    a, b = np.asarray(system, float), np.asarray(reference, float)
    d, m = a - b, (a + b) / 2
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    reg = stats.linregress(m, d)
    return BlandAltman(bias, sd, (bias - 1.96 * sd, bias + 1.96 * sd), float(reg.slope), float(reg.pvalue), len(d))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))
