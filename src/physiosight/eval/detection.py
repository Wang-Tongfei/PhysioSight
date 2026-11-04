"""Per-repetition fault detection metrics (precision / recall / F1 per fault code)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int


def per_rep_prf(pred: list[set[str]], gold: list[set[str]]) -> dict[str, PRF]:
    """``pred[i]``/``gold[i]``: fault codes of repetition i (aligned by index)."""
    if len(pred) != len(gold):
        raise ValueError("pred and gold must be aligned per repetition")
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for p, g in zip(pred, gold, strict=True):
        for c in p & g:
            tp[c] += 1
        for c in p - g:
            fp[c] += 1
        for c in g - p:
            fn[c] += 1
    out = {}
    for c in sorted(set(tp) | set(fp) | set(fn)):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = PRF(prec, rec, f1, tp[c] + fn[c])
    return out
