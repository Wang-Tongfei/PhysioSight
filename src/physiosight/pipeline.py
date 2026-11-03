"""Wiring: source -> pose backend -> smoothing -> analyser -> events -> Session."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from physiosight.events import EventBus
from physiosight.exercises.machines import Analyzer, FrameResult, create_analyzer
from physiosight.exercises.spec import ExerciseSpec, load_spec
from physiosight.perception.base import Observation, PoseFrame
from physiosight.perception.cache import LandmarkSequence
from physiosight.session.builder import SessionBuilder
from physiosight.session.models import Session, SourceInfo
from physiosight.signal.filters import LandmarkSmoother


@dataclass
class AnalysisRun:
    session: Session
    frames: list[FrameResult] = field(default_factory=list)


class Analysis:
    """Stateful per-session analysis usable both offline and live."""

    def __init__(self, spec: ExerciseSpec, target: float, *, source: SourceInfo,
                 pose_backend: str, pose_backend_version: str | None = None,
                 params: Mapping[str, float] | None = None, spec_sha256: str | None = None,
                 smoother: LandmarkSmoother | None = None, stop_at_target: bool = True,
                 bus: EventBus | None = None, skeleton: str | None = None) -> None:
        self.analyzer: Analyzer = create_analyzer(spec, target, params, skeleton)
        self.smoother = smoother
        self.stop_at_target = stop_at_target
        self.bus = bus or EventBus()
        self.builder = SessionBuilder(self.analyzer, source, pose_backend, pose_backend_version,
                                      spec_sha256, smoother.description if smoother else None)
        self.done = False
        self._last_t = 0.0

    def step(self, t: float, pose: PoseFrame | None) -> FrameResult | None:
        if self.done:
            return None
        if self.smoother is not None:
            pose = self.smoother(pose, t)
        result = self.analyzer.update(Observation(t, pose))
        self._last_t = t
        self.builder.add(result)
        for ev in result.events:
            self.bus.publish(ev)
        if self.stop_at_target and self.analyzer.is_complete:
            self.done = True
        return result

    def finish(self) -> Session:
        for ev in self.analyzer.finish(self._last_t):
            self.bus.publish(ev)
        return self.builder.build()


def analyze_sequence(seq: LandmarkSequence, exercise: str | ExerciseSpec, target: float, *,
                     side: str | None = None, params: Mapping[str, float] | None = None,
                     smoother: LandmarkSmoother | None = None, stop_at_target: bool = True,
                     keep_frames: bool = False,
                     on_frame: Callable[[FrameResult], None] | None = None) -> AnalysisRun:
    if isinstance(exercise, str):
        spec, sha = load_spec(exercise, side)
    else:
        spec, sha = exercise, None
    source = SourceInfo(kind="synthetic" if seq.meta.get("synthetic") else "landmarks",
                        width=seq.width, height=seq.height, nominal_fps=seq.nominal_fps,
                        mirrored=bool(seq.meta.get("mirrored", False)))
    analysis = Analysis(spec, target, source=source,
                        pose_backend=str(seq.meta.get("pose_backend", "cache")),
                        pose_backend_version=seq.meta.get("pose_backend_version"),
                        params=params, spec_sha256=sha, smoother=smoother,
                        stop_at_target=stop_at_target, skeleton=seq.skeleton)
    frames: list[FrameResult] = []
    for obs in seq.observations():
        res = analysis.step(obs.t, obs.pose)
        if res is None:
            break
        if keep_frames:
            frames.append(res)
        if on_frame:
            on_frame(res)
    return AnalysisRun(analysis.finish(), frames)


def extract_landmarks(source, backend, limit_s: float | None = None) -> LandmarkSequence:
    """Run a pose backend over a frame source and cache the result."""
    frames: list[tuple[float, PoseFrame | None]] = []
    for t, frame in source.frames():
        if limit_s is not None and t > limit_s:
            break
        frames.append((t, backend.process(frame, t)))
    meta = {"pose_backend": backend.name, "pose_backend_version": backend.version,
            "source": getattr(source, "label", None), "mirrored": False}
    return LandmarkSequence.from_frames(frames, backend.skeleton, source.width, source.height, meta)


def iter_frames(seq: LandmarkSequence) -> Iterable[Observation]:
    return seq.observations()
