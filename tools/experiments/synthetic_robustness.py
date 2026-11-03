"""Synthetic robustness study of the counting / fault logic (NOT of pose accuracy).

Randomised scripts with known ground truth are rendered over a grid of
frame rates {10, 15, 30, 60}, keypoint noise {0, 2, 4} px, aspect ratios
{16:9, 1:1} and smoothing {off, One Euro}; the new engine and the legacy
engine (replayed on identical landmarks) are scored with the same metrics.

    python -m tools.experiments.synthetic_robustness --n 30 -o docs/results/synthetic_robustness

Metrics per cell:
  exact / OBO / MAE      - valid-rep count vs ground truth (OBO = |error| <= 1)
  fault_recall           - faulty attempts flagged with the intended fault code
  false_alarm            - clean attempts that received any fault
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physiosight.eval.counting import exact, mae_absolute, obo
from physiosight.pipeline import analyze_sequence
from physiosight.signal.filters import LandmarkSmoother
from physiosight.testing import synth
from tools.baseline.legacy_adapter import legacy_available, replay


def random_script(ex: str, rng: np.random.Generator) -> tuple[synth.Script, str | None]:
    reps = int(rng.integers(2, 6))
    kind = str(rng.choice(["good", "good", "shallow", "fast", "form"]))
    pace = float(rng.uniform(0.6, 1.4))
    pause = float(rng.uniform(0.15, 0.8))  # short pauses stress the stability timers
    if ex == "pushup" and kind == "form":
        kind = "good"
    table = {
        "squat": {
            "good": (lambda: synth.squat_script(reps, depth=float(rng.uniform(70, 92)), down_s=pace, up_s=pace, pause_s=pause), None),
            "shallow": (lambda: synth.squat_script(reps, depth=float(rng.uniform(110, 140)), down_s=pace, up_s=pace, pause_s=pause), "shallow_squat"),
            "fast": (lambda: synth.squat_script(reps, down_s=0.2, up_s=0.2, pause_s=pause), "movement_too_fast"),
            "form": (lambda: synth.squat_script(reps, depth=70, asym=float(rng.uniform(25, 29)), down_s=pace, up_s=pace, pause_s=pause), "uneven_squat"),
        },
        "bicep_curl": {
            "good": (lambda: synth.curl_script(reps, top=float(rng.uniform(25, 38)), up_s=pace, down_s=pace, pause_s=pause), None),
            "shallow": (lambda: synth.curl_script(reps, top=float(rng.uniform(65, 100)), up_s=pace, down_s=pace, pause_s=pause), "partial_curl"),
            "fast": (lambda: synth.curl_script(reps, up_s=0.08, down_s=0.08, pause_s=pause), "movement_too_fast"),
            "form": (lambda: synth.curl_script(reps, drift=float(rng.uniform(45, 60)), up_s=pace, down_s=pace, pause_s=pause), "upper_arm_drift"),
        },
        "pushup": {
            "good": (lambda: synth.pushup_script(reps, bottom=float(rng.uniform(70, 92)), down_s=pace, up_s=pace, pause_s=pause), None),
            "shallow": (lambda: synth.pushup_script(reps, bottom=float(rng.uniform(115, 140)), down_s=pace, up_s=pace, pause_s=pause), "shallow_pushup"),
            "fast": (lambda: synth.pushup_script(reps, down_s=0.2, up_s=0.2, pause_s=pause), "movement_too_fast"),
        },
    }
    make, fault = table[ex][kind]
    sc = make()
    return sc, fault


def _attempts(sc: synth.Script) -> int:
    return (len(sc.moves) - 1) // 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="scripts per exercise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, default=Path("docs/results/synthetic_robustness"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    with_legacy = legacy_available()
    rows = []
    for ex in ("squat", "bicep_curl", "pushup"):
        scripts = [random_script(ex, rng) for _ in range(args.n)]
        for fps in (10, 15, 30, 60):
            for noise in (0.0, 2.0, 4.0):
                for size in ((1280, 720), (720, 720)):
                    for smooth in (False, True):
                        pred, gold, legacy = [], [], []
                        hit = faulty = alarm = clean = 0
                        for i, (sc, fault) in enumerate(scripts):
                            seq = synth.render(sc, fps=fps, size=size, noise_px=noise, seed=i)
                            s = analyze_sequence(seq, ex, 1000, stop_at_target=False,
                                                 smoother=LandmarkSmoother() if smooth else None).session
                            pred.append(s.summary.achieved)
                            gold.append(sc.expected_valid if fault is None else 0)
                            n_att = _attempts(sc)
                            rep_faults = [set(r.faults) for r in s.reps[:n_att]]
                            rep_faults += [set()] * (n_att - len(rep_faults))
                            if fault is None:
                                clean += n_att
                                alarm += sum(bool(f) for f in rep_faults)
                            else:
                                faulty += n_att
                                hit += sum(fault in f for f in rep_faults)
                            if with_legacy and not smooth:
                                legacy.append(replay(seq, ex, 1000, stop_at_target=False)[-1].progress)
                        row = {"exercise": ex, "fps": fps, "noise_px": noise, "aspect": f"{size[0]}x{size[1]}",
                               "smoothing": smooth, "n_scripts": len(scripts),
                               "exact": exact(pred, gold), "obo": obo(pred, gold), "mae": mae_absolute(pred, gold),
                               "fault_recall": hit / faulty if faulty else None,
                               "false_alarm": alarm / clean if clean else None}
                        if legacy:
                            row |= {"legacy_exact": exact(legacy, gold), "legacy_mae": mae_absolute(legacy, gold)}
                        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(rows, args) + "\n")
    print(_markdown(rows, args))
    return 0


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _markdown(rows, args) -> str:
    def table(group, title, filt=lambda r: True):
        g = defaultdict(list)
        for r in rows:
            if filt(r):
                g[group(r)].append(r)
        out = [f"### {title}", "", "| group | exact (new) | exact (legacy) | MAE (new) | MAE (legacy) | fault recall | false alarm |",
               "|---|---|---|---|---|---|---|"]
        for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])):
            out.append(f"| {k} | {_mean(rs, 'exact'):.3f} | {_mean(rs, 'legacy_exact'):.3f} | {_mean(rs, 'mae'):.2f} | "
                       f"{_mean(rs, 'legacy_mae'):.2f} | {_mean(rs, 'fault_recall'):.3f} | {_mean(rs, 'false_alarm'):.3f} |")
        return "\n".join(out)

    raw = lambda r: not r["smoothing"]  # noqa: E731
    return "\n\n".join([
        f"# Synthetic robustness (n={args.n} scripts per exercise, seed={args.seed})",
        "Synthetic stick-figure landmarks with known ground truth; measures the analysis logic only, "
        "not pose-model accuracy. Legacy columns replay the unmodified legacy engine on identical landmarks "
        "(no smoothing). Generated by `python -m tools.experiments.synthetic_robustness`.",
        table(lambda r: r["exercise"], "By exercise (no smoothing)", raw),
        table(lambda r: r["fps"], "By frame rate (no smoothing)", raw),
        table(lambda r: r["aspect"], "By aspect ratio (no smoothing)", raw),
        table(lambda r: (r["noise_px"], "1€" if r["smoothing"] else "raw"), "By keypoint noise x smoothing"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
