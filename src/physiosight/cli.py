"""Command line interface.

    physiosight specs                              list built-in exercises
    physiosight schema -o session.schema.json      export the Session JSON Schema
    physiosight extract VIDEO -o seq.npz           run the pose model once, cache landmarks
    physiosight analyze seq.npz -e squat -o s.json analyse cached landmarks (no CV deps)
    physiosight run [CAMERA|VIDEO] -e squat        interactive monitor (successor of physio_form_monitor.py)
    physiosight notify-test [--video clip.mp4]     check Telegram credentials
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
DEFAULT_HAND_MODEL = MODELS_DIR / "hand_landmarker.task"


def _load_overrides(path: str | None, pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if path:
        out.update({k: float(v) for k, v in json.loads(Path(path).read_text()).items()})
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = float(value)
    return out


def _smoother(args):
    if not args.smooth:
        return None
    from physiosight.signal.filters import LandmarkSmoother

    return LandmarkSmoother(min_cutoff=args.min_cutoff, beta=args.beta)


def cmd_specs(_args) -> int:
    from physiosight.exercises.spec import list_builtin_specs, load_spec

    for name in list_builtin_specs():
        spec, _ = load_spec(name)
        print(f"{spec.id:12s} {spec.kind:10s} v{spec.version}  sides={spec.sides}  view={spec.view}")
    return 0


def cmd_schema(args) -> int:
    from physiosight.session.models import Session

    text = json.dumps(Session.model_json_schema(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


def cmd_extract(args) -> int:
    from physiosight.capture.sources import VideoFileSource
    from physiosight.perception.mediapipe_backend import MediaPipePoseBackend
    from physiosight.pipeline import extract_landmarks

    source = VideoFileSource(args.video)
    backend = MediaPipePoseBackend(str(args.model))
    try:
        seq = extract_landmarks(source, backend, args.limit)
    finally:
        source.close()
        backend.close()
    seq.save(args.output)
    print(f"{len(seq)} frames, pose in {int(seq.present.sum())}, saved {args.output}")
    return 0


def cmd_analyze(args) -> int:
    from physiosight import i18n
    from physiosight.exercises.spec import load_spec
    from physiosight.perception.cache import LandmarkSequence
    from physiosight.pipeline import analyze_sequence

    seq = LandmarkSequence.load(args.landmarks)
    target = args.target or load_spec(args.exercise, args.side)[0].default_target
    run = analyze_sequence(seq, args.exercise, target, side=args.side,
                           params=_load_overrides(args.params, args.set), smoother=_smoother(args),
                           stop_at_target=not args.full)
    s = run.session
    text = s.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    unit = "reps" if s.exercise_kind == "repetition" else "s"
    print(f"{i18n.text('exercise.' + s.exercise_id, args.lang)}: {s.summary.achieved:g}/{s.summary.target:g} {unit}"
          f", attempts={s.summary.attempts}, faults={[f.code for f in s.faults]}", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    from physiosight.live import LiveMonitor, LiveOptions

    opts = LiveOptions(
        source=args.source, exercise=args.exercise, side=args.side, target=args.target,
        params=_load_overrides(args.params, args.set) or None, pose_model=str(args.model),
        hand_model=None if args.no_hands else str(args.hand_model), smoothing=args.smooth,
        clips_dir=args.clips or None, sessions_dir=args.sessions_dir, headless=args.headless,
        cover_face=not args.show_face, lang=args.lang,
    )
    LiveMonitor(opts).run()
    return 0


def cmd_notify_test(args) -> int:
    """Synchronously send a Telegram test message (and optionally a clip), printing the API response."""
    import os

    import requests

    token = os.environ.get("PHYSIOSIGHT_TG_TOKEN") or os.environ.get("PHYSIO_TG_TOKEN", "")
    chat = os.environ.get("PHYSIOSIGHT_TG_CHAT") or os.environ.get("PHYSIO_TG_CHAT", "")
    if not token or not chat:
        print("Set PHYSIOSIGHT_TG_TOKEN and PHYSIOSIGHT_TG_CHAT first.", file=sys.stderr)
        return 2
    base = f"https://api.telegram.org/bot{token}"
    r = requests.post(f"{base}/sendMessage", data={"chat_id": chat, "text": "PhysioSight: test alert"}, timeout=30)
    print("sendMessage:", r.status_code, r.text[:200])
    if r.ok and args.video:
        with open(args.video, "rb") as fh:
            r = requests.post(f"{base}/sendVideo", data={"chat_id": chat, "caption": "PhysioSight: test clip"},
                              files={"video": (Path(args.video).name, fh, "video/mp4")}, timeout=120)
        print("sendVideo:", r.status_code, r.text[:300])
    if not r.ok:
        print("401 = wrong token, 400 'chat not found' = wrong chat id, 403 = press START on the bot first, "
              "413 = clip too large.", file=sys.stderr)
    return 0 if r.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="physiosight")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("specs").set_defaults(fn=cmd_specs)
    sp = sub.add_parser("schema")
    sp.add_argument("-o", "--output")
    sp.set_defaults(fn=cmd_schema)

    sp = sub.add_parser("extract")
    sp.add_argument("video")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--model", default=DEFAULT_MODEL)
    sp.add_argument("--limit", type=float, help="seconds")
    sp.set_defaults(fn=cmd_extract)

    def analysis_args(sp):
        sp.add_argument("-e", "--exercise", default="bicep_curl", help="built-in id or path to YAML")
        sp.add_argument("--side")
        sp.add_argument("--target", type=float, help="reps or seconds (default: the spec's default_target)")
        sp.add_argument("--params", help="JSON file of threshold overrides (calibration)")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        sp.add_argument("--smooth", action="store_true",
                        help="One Euro keypoint smoothing (off by default, see docs/DECISIONS.md D12)")
        sp.add_argument("--min-cutoff", type=float, default=1.0)
        sp.add_argument("--beta", type=float, default=0.01)
        sp.add_argument("--lang", default="en")
        sp.add_argument("-o", "--output")

    sp = sub.add_parser("analyze")
    sp.add_argument("landmarks")
    sp.add_argument("--full", action="store_true", help="keep analysing after the target is reached")
    analysis_args(sp)

    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("run", help="interactive monitor on a camera or video")
    sp.add_argument("source", nargs="?", default="0", help="camera index (default 0) or video path")
    sp.add_argument("--model", default=DEFAULT_MODEL)
    sp.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    sp.add_argument("--no-hands", action="store_true")
    sp.add_argument("--clips", default="clips", help="fault clip directory ('' to disable)")
    sp.add_argument("--sessions-dir", help="write one Session JSON per exercise segment")
    sp.add_argument("--headless", action="store_true", help="no window (video files only)")
    sp.add_argument("--show-face", action="store_true", help="do not cover the face in view/clips")
    analysis_args(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("notify-test", help="check Telegram credentials")
    sp.add_argument("--video", help="also send this clip")
    sp.set_defaults(fn=cmd_notify_test)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
