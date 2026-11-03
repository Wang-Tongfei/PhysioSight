"""Collects analyser output into a :class:`Session`."""
from __future__ import annotations

from typing import Any

from physiosight import NAME, __version__
from physiosight.events import FaultDetected
from physiosight.exercises.machines import Analyzer, FrameResult, HoldAnalyzer, RepetitionAnalyzer
from physiosight.exercises.spec import HoldSpec
from physiosight.session.models import (
    AngleSpace,
    EngineInfo,
    ExerciseKind,
    FaultEvent,
    Session,
    SessionSummary,
    SourceInfo,
    TrackingQuality,
)


class SessionBuilder:
    def __init__(
        self,
        analyzer: Analyzer,
        source: SourceInfo,
        pose_backend: str,
        pose_backend_version: str | None = None,
        spec_sha256: str | None = None,
        smoothing: str | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.source = source
        self.pose_backend = pose_backend
        self.pose_backend_version = pose_backend_version
        self.spec_sha256 = spec_sha256
        self.smoothing = smoothing
        self.faults: list[FaultEvent] = []
        self.first_t: float | None = None
        self.last_t = 0.0

    def add(self, result: FrameResult) -> None:
        if self.first_t is None:
            self.first_t = result.t
        self.last_t = result.t
        for ev in result.events:
            if isinstance(ev, FaultDetected):
                self.faults.append(FaultEvent(t_s=max(0.0, ev.t - self.first_t),
                                              code=ev.code, rep_index=ev.rep_index))

    def build(self) -> Session:
        a = self.analyzer
        spec = a.spec
        t0 = self.first_t or 0.0
        shift = lambda x: max(0.0, x - t0)  # noqa: E731
        duration = shift(self.last_t)
        uses_world = any(getattr(f, "space", "image") == "world" for f in spec.features.values())
        engine = EngineInfo(
            name=NAME, version=__version__, pose_backend=self.pose_backend,
            pose_backend_version=self.pose_backend_version,
            angle_space=AngleSpace.WORLD_3D if uses_world else AngleSpace.IMAGE_2D,
            smoothing=self.smoothing, spec_id=spec.id, spec_version=spec.version,
            spec_sha256=self.spec_sha256, params=dict(a.params),
        )
        tracking = TrackingQuality(
            frames_total=a.frames_total, frames_with_pose=a.frames_with_pose,
            frames_reliable=a.frames_reliable, tracking_lost_events=a.tracking_lost_events,
            frames_view_ok=a.frames_view_ok,
        )
        src = self.source
        if duration > 0 and a.frames_total > 1:
            src = src.model_copy(update={"effective_fps": (a.frames_total - 1) / duration})
        side = spec.sides[0]
        common: dict[str, Any] = dict(exercise_id=spec.id, side=side, source=src, engine=engine,
                      tracking=tracking, faults=self.faults)
        if isinstance(spec, HoldSpec):
            assert isinstance(a, HoldAnalyzer)
            holds = [h.model_copy(update={"started_at_s": shift(h.started_at_s),
                                          "ended_at_s": shift(h.ended_at_s)}) for h in a.holds]
            return Session(
                exercise_kind=ExerciseKind.HOLD, holds=holds,
                summary=SessionSummary(target=a.target, achieved=round(a.credited, 6),
                                       attempts=len(holds), completed=a.is_complete,
                                       duration_s=duration),
                **common,
            )
        assert isinstance(a, RepetitionAnalyzer)
        reps = [r.model_copy(update={"started_at_s": shift(r.started_at_s),
                                     "ended_at_s": shift(r.ended_at_s)}) for r in a.reps]
        return Session(
            exercise_kind=ExerciseKind.REPETITION, reps=reps,
            summary=SessionSummary(target=a.target, achieved=a.progress, attempts=len(reps),
                                   completed=a.is_complete, duration_s=duration),
            **common,
        )
