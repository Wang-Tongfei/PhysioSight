"""Validate PhysioSight on your own recordings (run locally; needs video + labels).

1) Counting / fault detection against hand labels::

    python tools/validate/evaluate.py counts labels.csv --cache cache/ -o report_counts.md

   labels.csv columns: video,exercise,side,true_reps[,true_faults]
   (true_faults: ';'-separated fault codes that occurred at least once)

2) Joint-angle agreement against a reference system (goniometer, IMU, OpenCap,
   optical mocap)::

    python tools/validate/evaluate.py angles reference.csv --sessions sessions/ -o report_angles.md

   reference.csv columns: session_file,rep_index,angle,reference_deg[,stat]
   (stat = max_deg|min_deg, default max_deg; angle e.g. knee_flexion_left)

Reports ICC(2,1) with 95% CI (Koo & Li 2016 interpretation), Bland-Altman
bias / limits of agreement, RMSE and MAE. Split by participant when you
use these numbers to tune thresholds - never tune and report on the same people.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiosight.eval.counting import exact, mae_absolute, mae_normalized, obo  # noqa: E402
from physiosight.exercises.spec import load_spec  # noqa: E402
from physiosight.perception.cache import LandmarkSequence  # noqa: E402
from physiosight.pipeline import analyze_sequence  # noqa: E402
from physiosight.session.models import Session  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_lite.task"


def _landmarks(video: Path, cache: Path, model: Path) -> LandmarkSequence:
    cached = cache / f"{video.stem}.npz"
    if cached.exists():
        return LandmarkSequence.load(cached)
    from physiosight.capture.sources import VideoFileSource
    from physiosight.perception.mediapipe_backend import MediaPipePoseBackend
    from physiosight.pipeline import extract_landmarks

    src, backend = VideoFileSource(str(video)), MediaPipePoseBackend(str(model))
    try:
        seq = extract_landmarks(src, backend)
    finally:
        src.close()
        backend.close()
    cache.mkdir(parents=True, exist_ok=True)
    seq.save(cached)
    return seq


def cmd_counts(args) -> int:
    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    pred, gold, lines = [], [], []
    tp = fp = fn = 0
    for r in rows:
        video = (Path(args.labels).parent / r["video"]).resolve()
        side = r.get("side") or None
        spec, _ = load_spec(r["exercise"], side if side in ("left", "right") else None)
        seq = _landmarks(video, Path(args.cache), Path(args.model))
        s = analyze_sequence(seq, spec, 1000, stop_at_target=False).session
        if args.sessions_out:
            Path(args.sessions_out).mkdir(parents=True, exist_ok=True)
            (Path(args.sessions_out) / f"{video.stem}.json").write_text(s.model_dump_json(indent=2))
        true_reps = float(r["true_reps"])
        pred.append(s.summary.achieved)
        gold.append(true_reps)
        got = {f.code for f in s.faults}
        want = {c for c in (r.get("true_faults") or "").split(";") if c}
        tp, fp, fn = tp + len(got & want), fp + len(got - want), fn + len(want - got)
        lines.append(f"| {video.name} | {r['exercise']} | {true_reps:g} | {s.summary.achieved:g} | "
                     f"{s.tracking.reliable_ratio:.2f} | {';'.join(sorted(want))} | {';'.join(sorted(got))} |")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    report = ["# Counting validation", "",
              f"videos: {len(rows)}  |  exact: {exact(pred, gold):.2f}  |  OBO: {obo(pred, gold):.2f}  |  "
              f"MAE: {mae_absolute(pred, gold):.2f} reps  |  normalised MAE: "
              f"{mae_normalized(pred, gold) if min(gold) > 0 else float('nan'):.3f}",
              f"fault codes (session level): precision {prec:.2f}, recall {rec:.2f}", "",
              "| video | exercise | true | predicted | reliable frames | true faults | detected faults |",
              "|---|---|---|---|---|---|---|", *lines]
    Path(args.output).write_text("\n".join(report) + "\n")
    print("\n".join(report[:4]))
    return 0


def cmd_angles(args) -> int:
    from physiosight.eval.agreement import bland_altman, icc, interpret_icc, mae, rmse

    rows = list(csv.DictReader(open(args.reference, newline="", encoding="utf-8")))
    sessions: dict[str, Session] = {}
    by_angle: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        f = r["session_file"]
        if f not in sessions:
            sessions[f] = Session.model_validate_json((Path(args.sessions) / f).read_text())
        rep = next(x for x in sessions[f].reps if x.index == int(r["rep_index"]))
        ext = rep.angles.get(r["angle"])
        if ext is None:
            continue
        value = getattr(ext, r.get("stat") or "max_deg")
        by_angle.setdefault(r["angle"], []).append((value, float(r["reference_deg"])))
    out = ["# Angle agreement", "", "| angle | n | ICC(2,1) [95% CI] | band | bias | LoA | RMSE | MAE |",
           "|---|---|---|---|---|---|---|---|"]
    for name, pairs in sorted(by_angle.items()):
        a = np.array(pairs)
        if len(a) < 3:
            continue
        ic, ba = icc(a, "2,1"), bland_altman(a[:, 0], a[:, 1])
        out.append(f"| {name} | {len(a)} | {ic.value:.2f} [{ic.ci95[0]:.2f}, {ic.ci95[1]:.2f}] | {interpret_icc(ic.value)} | "
                   f"{ba.bias:+.1f} | [{ba.loa[0]:.1f}, {ba.loa[1]:.1f}] | {rmse(a[:, 0], a[:, 1]):.1f} | "
                   f"{mae(a[:, 0], a[:, 1]):.1f} |")
    Path(args.output).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counts")
    c.add_argument("labels")
    c.add_argument("--cache", default="cache")
    c.add_argument("--model", default=str(DEFAULT_MODEL))
    c.add_argument("--sessions-out")
    c.add_argument("-o", "--output", default="report_counts.md")
    c.set_defaults(fn=cmd_counts)
    a = sub.add_parser("angles")
    a.add_argument("reference")
    a.add_argument("--sessions", required=True)
    a.add_argument("-o", "--output", default="report_angles.md")
    a.set_defaults(fn=cmd_angles)
    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
