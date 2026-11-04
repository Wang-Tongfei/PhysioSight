"""Generic state machines that execute exercise specs.

Only two machines exist:

* :class:`RepetitionAnalyzer` - ready -> start -> target -> return, with
  checks that invalidate the attempt and a minimum repetition duration.
* :class:`HoldAnalyzer` - accumulates time while a validity condition holds.

All stability requirements are expressed in **seconds** (``hold_s``), using
the frame timestamps, so behaviour does not change with frame rate. A
condition is "held" once it has been continuously true for ``hold_s``
seconds, i.e. the legacy "N consecutive frames at 30 fps" equals
``hold_s = (N - 1) / 30``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from physiosight.events import (
    Event,
    FaultDetected,
    HoldSegmentClosed,
    RepFinished,
    RepStarted,
    TargetReached,
    TrackingLost,
    TrackingRestored,
)
from physiosight.exercises.expr import Expr
from physiosight.exercises.spec import (
    ExerciseSpec,
    HoldSpec,
    RepetitionSpec,
    eval_number,
    resolve_params,
)
from physiosight.kinematics.features import CompiledFeatures
from physiosight.perception.base import SKELETONS, Observation
from physiosight.session.models import AngleExtrema, HoldSegment, RepOutcome, RepRecord

EPS = 1e-6


@dataclass(frozen=True, slots=True)
class FrameResult:
    t: float
    tracking_ok: bool
    form_ok: bool
    progress: float
    hint_key: str
    metric_label_key: str | None = None
    metric_value: float | None = None
    phase: str = "idle"
    events: tuple[Event, ...] = ()


class _Timer:
    """Tracks how long a condition has been continuously true."""

    __slots__ = ("since",)

    def __init__(self) -> None:
        self.since: float | None = None

    def update(self, value: bool, t: float) -> None:
        if not value:
            self.since = None
        elif self.since is None:
            self.since = t

    def held(self, t: float, hold_s: float) -> bool:
        return self.since is not None and t - self.since >= hold_s - EPS

    def reset(self) -> None:
        self.since = None


class _Base:
    spec: ExerciseSpec

    def __init__(
        self,
        spec: ExerciseSpec,
        target: float,
        params: Mapping[str, float] | None = None,
        skeleton: str | None = None,
    ) -> None:
        self.spec = spec
        self.skeleton = skeleton or spec.skeleton
        self.target = float(target)
        self.params = resolve_params(spec, params) if params is not None else resolve_params(spec)
        self._penv = {f"p.{k}": v for k, v in self.params.items()}
        # specs name keypoints, so the same spec runs on any skeleton that has them
        self._features = CompiledFeatures.compile(spec.features, self.skeleton)
        names = SKELETONS[self.skeleton]
        missing = [n for n in spec.required if n not in names]
        if missing:
            raise ValueError(f"{spec.id}: required keypoints {missing} not in skeleton {self.skeleton}")
        self._required = [names.index(n) for n in spec.required]
        self._derived = [(k, Expr.parse(v)) for k, v in spec.derived.items()]
        self._checks = [(c, Expr.parse(c.when)) for c in spec.checks]
        self._hints = [(h.key, Expr.parse(h.when)) for h in spec.hints]
        self._metric = Expr.parse(spec.primary_metric.expr) if spec.primary_metric else None
        self._grace = eval_number(spec.tracking_grace_s, self.params) or 0.0
        self._view = Expr.parse(spec.view_check) if spec.view_check else None
        self.frames_view_ok: int | None = 0 if self._view else None
        self.view_ok_now = True
        self._check_active = {c.id: False for c in spec.checks}
        self.frames_total = 0
        self.frames_with_pose = 0
        self.frames_reliable = 0
        self.tracking_lost_events = 0
        self._tracking_ok = True
        self._invalid_since: float | None = None

    # ---------------------------------------------------------------- env --
    def _env(self, obs: Observation) -> dict[str, float] | None:
        self.frames_total += 1
        pose = obs.pose
        if pose is None:
            return None
        self.frames_with_pose += 1
        conf = self.spec.confidence
        if not pose.reliable(self._required, conf.min_visibility, conf.min_presence):
            return None
        self.frames_reliable += 1
        env: dict[str, float] = dict(self._penv)
        env.update(self._features.compute(pose))
        for name, expr in self._derived:
            env[name] = expr.evaluate(env)
        if self._view is not None:
            self.view_ok_now = self._view.truth(env)
            self.frames_view_ok = (self.frames_view_ok or 0) + int(self.view_ok_now)
        return env

    def _tracking_events(self, ok: bool, t: float, events: list[Event]) -> None:
        if ok and not self._tracking_ok:
            events.append(TrackingRestored(t))
        elif not ok and self._tracking_ok:
            events.append(TrackingLost(t))
            self.tracking_lost_events += 1
        self._tracking_ok = ok

    def _hint(self, env: Mapping[str, float], state: Mapping[str, float]) -> str:
        full = {**env, **state}
        for key, expr in self._hints:
            if expr.truth(full):
                return key
        return self.spec.default_hint

    def _metric_value(self, env: Mapping[str, float]) -> float | None:
        if self._metric is None:
            return None
        v = self._metric.evaluate(env)
        return None if math.isnan(v) else v


class RepetitionAnalyzer(_Base):
    spec: RepetitionSpec

    def __init__(self, spec: RepetitionSpec, target: float, params: Mapping[str, float] | None = None,
                 skeleton: str | None = None):
        super().__init__(spec, target, params, skeleton)
        p = self.params
        self._phase_expr = {n: Expr.parse(getattr(spec, n).when) for n in ("ready", "start", "target", "finish")}
        self._phase_hold = {n: eval_number(getattr(spec, n).hold_s, p) or 0.0 for n in self._phase_expr}
        self._timers = {n: _Timer() for n in self._phase_expr}
        self._min_dur = eval_number(spec.min_rep_duration_s, p) or 0.0
        self._record = [(k, Expr.parse(v)) for k, v in spec.record.items()]
        self.progress = 0
        self.attempts = 0
        self.reps: list[RepRecord] = []
        self._reset_attempt(full=True)

    @property
    def is_complete(self) -> bool:
        return self.progress >= self.target

    def _reset_attempt(self, full: bool) -> None:
        self.in_movement = False
        self.at_target = False
        self._started_at = 0.0
        self._rep_valid = True
        self._rep_faults: list[str] = []
        self._extrema: dict[str, list[float]] = {}
        self._view_frames = [0, 0]  # ok, total
        if full:
            self.ready = False
            for timer in self._timers.values():
                timer.reset()
            self._check_active = {k: False for k in self._check_active}

    def _close_attempt(self, t: float, outcome: RepOutcome, faults: list[str]) -> RepRecord:
        record = RepRecord(
            index=self.attempts,
            started_at_s=self._started_at,
            ended_at_s=max(t, self._started_at),
            outcome=outcome,
            reached_target=self.at_target,
            faults=[] if outcome is RepOutcome.VALID else list(dict.fromkeys(faults)),
            angles={k: AngleExtrema(min_deg=lo, max_deg=hi) for k, (lo, hi) in self._extrema.items()},
            view_ok=(self._view_frames[0] >= 0.8 * self._view_frames[1])
            if self._view is not None and self._view_frames[1] else None,
        )
        self.reps.append(record)
        return record

    def update(self, obs: Observation) -> FrameResult:
        t = obs.t
        events: list[Event] = []
        env = self._env(obs)
        if env is None:
            if self._invalid_since is None:
                self._invalid_since = t
            if t - self._invalid_since >= self._grace - EPS:
                self._tracking_events(False, t, events)
                if self.in_movement:
                    rec = self._close_attempt(t, RepOutcome.ABORTED, ["tracking_lost"])
                    events.append(RepFinished(t, rec))
                self._reset_attempt(full=True)
            return FrameResult(t, False, False, float(self.progress), self.spec.tracking_hint,
                               phase="tracking_lost", events=tuple(events))
        self._invalid_since = None
        self._tracking_events(True, t, events)

        truth = {n: e.truth(env) for n, e in self._phase_expr.items()}
        for n, timer in self._timers.items():
            timer.update(truth[n], t)
        check_true = {c.id: expr.truth(env) for c, expr in self._checks}

        if not self.in_movement and self._timers["ready"].held(t, self._phase_hold["ready"]):
            self.ready = True
        if (not self.in_movement and self.ready
                and self._timers["start"].held(t, self._phase_hold["start"])):
            self.in_movement = True
            self.at_target = False
            self._started_at = t
            self._rep_valid = True
            self._rep_faults = []
            self._extrema = {}
            self._view_frames = [0, 0]
            self.attempts += 1
            events.append(RepStarted(t, self.attempts))

        if self.in_movement:
            self._view_frames[0] += int(self.view_ok_now)
            self._view_frames[1] += 1
            for c, _ in self._checks:
                if check_true[c.id] and c.invalidates_rep:
                    self._rep_valid = False
                    self._rep_faults.append(c.fault)
            for name, expr in self._record:
                v = expr.evaluate(env)
                if not math.isnan(v):
                    lo_hi = self._extrema.setdefault(name, [v, v])
                    lo_hi[0], lo_hi[1] = min(lo_hi[0], v), max(lo_hi[1], v)
            if self._timers["target"].held(t, self._phase_hold["target"]):
                self.at_target = True

        frame_fault = False
        if self.in_movement and self._timers["finish"].held(t, self._phase_hold["finish"]):
            duration = t - self._started_at
            too_fast = duration < self._min_dur - EPS
            if self.at_target and self._rep_valid and not too_fast:
                self.progress += 1
                rec = self._close_attempt(t, RepOutcome.VALID, [])
                if self.is_complete:
                    events.append(TargetReached(t))
            else:
                if too_fast:
                    end_fault = self.spec.too_fast_fault
                elif not self._rep_valid:
                    end_fault = self._rep_faults[0]
                else:
                    end_fault = self.spec.missed_target_fault
                rec = self._close_attempt(t, RepOutcome.INVALID, [end_fault, *self._rep_faults]
                                          + ([] if self.at_target else [self.spec.missed_target_fault]))
                events.append(FaultDetected(t, end_fault, rec.index))
                frame_fault = True
            events.append(RepFinished(t, rec))
            self._reset_attempt(full=False)
            self.ready = True

        for c, _ in self._checks:
            active = check_true[c.id] and (c.scope == "always" or self.in_movement)
            if active and not self._check_active[c.id] and not frame_fault:
                events.append(FaultDetected(t, c.fault, self.attempts if self.in_movement else None))
                frame_fault = True
            self._check_active[c.id] = active

        state = {
            "state.ready": float(self.ready), "state.in_movement": float(self.in_movement),
            "state.at_target": float(self.at_target), "state.tracking_ok": 1.0,
        }
        phase = ("at_target" if self.at_target else "moving") if self.in_movement else (
            "ready" if self.ready else "waiting")
        return FrameResult(
            t, True, not frame_fault and not any(check_true.values()), float(self.progress),
            self._hint(env, state),
            self.spec.primary_metric.label_key if self.spec.primary_metric else None,
            self._metric_value(env), phase, tuple(events),
        )

    def finish(self, t: float) -> list[Event]:
        """Close an attempt still in progress when the input ends."""
        if not self.in_movement:
            return []
        rec = self._close_attempt(t, RepOutcome.ABORTED, ["session_ended"])
        self._reset_attempt(full=True)
        return [RepFinished(t, rec)]


class HoldAnalyzer(_Base):
    spec: HoldSpec

    def __init__(self, spec: HoldSpec, target: float, params: Mapping[str, float] | None = None,
                 skeleton: str | None = None):
        super().__init__(spec, target, params, skeleton)
        p = self.params
        self._valid = Expr.parse(spec.valid_when)
        self._max_dt = eval_number(spec.max_dt_s, p) or math.inf
        self._reset_after = eval_number(spec.reset_after_break_s, p)
        self._min_segment = eval_number(spec.min_segment_s, p) or 0.0
        self.credited = 0.0
        self._broken_for = 0.0
        self._last_t: float | None = None
        self._segment_start: float | None = None
        self._segment_end = 0.0
        self.holds: list[HoldSegment] = []
        self._completed = False

    @property
    def progress(self) -> float:
        return self.credited

    @property
    def attempts(self) -> int:
        return len(self.holds)

    @property
    def is_complete(self) -> bool:
        return self.credited >= self.target - EPS

    def _close_segment(self, t: float, events: list[Event]) -> None:
        if self._segment_start is None:
            return
        seg = HoldSegment(started_at_s=self._segment_start, ended_at_s=self._segment_end)
        self._segment_start = None
        if seg.duration_s >= self._min_segment - EPS:
            self.holds.append(seg)
            events.append(HoldSegmentClosed(t, seg))

    def _break(self, elapsed: float) -> None:
        self._broken_for += elapsed
        if self._reset_after is not None and self._broken_for >= self._reset_after - EPS:
            self.credited = 0.0

    def update(self, obs: Observation) -> FrameResult:
        t = obs.t
        events: list[Event] = []
        elapsed = 0.0 if self._last_t is None else min(t - self._last_t, self._max_dt)
        self._last_t = t
        env = self._env(obs)
        if env is None:
            self._break(elapsed)
            self._close_segment(t, events)
            self._check_active = {k: False for k in self._check_active}
            if self._invalid_since is None:
                self._invalid_since = t
            if t - self._invalid_since >= self._grace - EPS:
                self._tracking_events(False, t, events)
            return FrameResult(t, False, False, self.credited, self.spec.tracking_hint,
                               phase="tracking_lost", events=tuple(events))
        self._invalid_since = None
        self._tracking_events(True, t, events)

        valid = self._valid.truth(env)
        if valid:
            self.credited = min(self.target, self.credited + elapsed)
            self._broken_for = 0.0
            if self._segment_start is None:
                self._segment_start = max(0.0, t - elapsed)
            self._segment_end = t
            if self.is_complete and not self._completed:
                self._completed = True
                events.append(TargetReached(t))
        else:
            self._break(elapsed)
            self._close_segment(t, events)

        frame_fault = False
        for c, expr in self._checks:
            active = expr.truth(env)
            if active and not self._check_active[c.id] and not frame_fault:
                events.append(FaultDetected(t, c.fault))
                frame_fault = True
            self._check_active[c.id] = active

        state = {"state.ready": 1.0, "state.in_movement": float(valid),
                 "state.at_target": float(self.is_complete), "state.tracking_ok": 1.0}
        return FrameResult(
            t, True, valid, self.credited, self._hint(env, state),
            self.spec.primary_metric.label_key if self.spec.primary_metric else None,
            self._metric_value(env), "holding" if valid else "waiting", tuple(events),
        )

    def finish(self, t: float) -> list[Event]:
        events: list[Event] = []
        self._close_segment(t, events)
        return events


Analyzer = RepetitionAnalyzer | HoldAnalyzer


def create_analyzer(
    spec: ExerciseSpec, target: float, params: Mapping[str, float] | None = None,
    skeleton: str | None = None,
) -> Analyzer:
    if isinstance(spec, RepetitionSpec):
        return RepetitionAnalyzer(spec, target, params, skeleton)
    return HoldAnalyzer(spec, target, params, skeleton)
