"""Create a baseline for the migration from the legacy engine.

For every video:  extract landmarks once (MediaPipe, video timestamps) ->
<out>/<name>.npz, replay them through the legacy engine -> <name>.legacy.jsonl
(per-frame trace), and through the new engine -> <name>.session.json.
A summary table compares the two.

    python tools/baseline/make_baseline.py videos/*.mp4 -e squat -o baseline/
    python tools/baseline/make_baseline.py baseline/*.npz -e squat -o baseline/   # reuse cache
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from physiosight.perception.cache import LandmarkSequence
from physiosight.pipeline import analyze_sequence
from tools.baseline.legacy_adapter import ROOT, replay  # noqa: F401  (sets sys.path)


def _landmarks(path: Path, out: Path, model: Path) -> LandmarkSequence:
    if path.suffix == ".npz":
        return LandmarkSequence.load(path)
    from physiosight.capture.sources import VideoFileSource
    from physiosight.perception.mediapipe_backend import MediaPipePoseBackend
    from physiosight.pipeline import extract_landmarks

    cached = out / f"{path.stem}.npz"
    if cached.exists():
        return LandmarkSequence.load(cached)
    src, backend = VideoFileSource(str(path)), MediaPipePoseBackend(str(model))
    try:
        seq = extract_landmarks(src, backend)
    finally:
        src.close()
        backend.close()
    seq.save(cached)
    return seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-e", "--exercise", required=True)
    ap.add_argument("--side", default="right")
    ap.add_argument("--target", type=float, default=10)
    ap.add_argument("-o", "--out", type=Path, default=Path("baseline"))
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "pose_landmarker_lite.task")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    side = args.side if args.exercise == "bicep_curl" else None
    rows = []
    for path in args.inputs:
        seq = _landmarks(path, args.out, args.model)
        legacy = replay(seq, args.exercise, args.target, arm=args.side)
        with open(args.out / f"{path.stem}.legacy.jsonl", "w") as fh:
            for fr in legacy:
                fh.write(json.dumps(asdict(fr)) + "\n")
        session = analyze_sequence(seq, args.exercise, args.target, side=side).session
        (args.out / f"{path.stem}.session.json").write_text(session.model_dump_json(indent=2))
        legacy_faults = [f.fault for f in legacy if f.fault]
        rows.append((path.stem, legacy[-1].progress if legacy else 0, session.summary.achieved,
                     len(legacy_faults), len(session.faults)))
    print(f"{'video':30s} {'legacy':>7s} {'new':>7s} {'faults(l/n)':>12s}")
    for name, lp, npg, lf, nf in rows:
        print(f"{name:30s} {lp:7g} {npg:7g} {lf:5d}/{nf:<6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
