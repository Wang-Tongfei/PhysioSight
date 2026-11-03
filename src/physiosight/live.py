"""Interactive monitor (camera or video) - the successor of physio_form_monitor.py.

Keys (click the window first): 1 curl, 2 squat, 3 plank, 4 push-up,
a switch side, r reset, q quit.

Every exercise segment (between start/switch/reset) becomes one Session. With
``sessions_dir`` set, each is written as JSON when the segment ends.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from physiosight import i18n
from physiosight.events import FaultDetected, RepFinished, TargetReached
from physiosight.exercises.spec import load_spec
from physiosight.perception.base import PoseFrame
from physiosight.pipeline import Analysis
from physiosight.plugins.clips import ClipRecorder
from physiosight.plugins.telegram import TelegramNotifier
from physiosight.session.models import Session, SourceInfo

log = logging.getLogger(__name__)
EXERCISE_KEYS = {ord("1"): "bicep_curl", ord("2"): "squat", ord("3"): "plank", ord("4"): "pushup"}


@dataclass
class LiveOptions:
    source: str
    exercise: str
    side: str | None = None
    target: float | None = None
    params: dict[str, float] | None = None
    pose_model: str = ""
    hand_model: str | None = None
    smoothing: bool = False
    clips_dir: str | None = None
    clip_cooldown_s: float = 15.0
    sessions_dir: str | None = None
    headless: bool = False
    cover_face: bool = True
    lang: str = "en"


def mirror_pose(pose: PoseFrame | None, width: int) -> PoseFrame | None:
    if pose is None:
        return None
    xy = pose.xy.copy()
    xy[:, 0] = width - xy[:, 0]
    return replace(pose, xy=xy)


class LiveMonitor:
    def __init__(self, opts: LiveOptions) -> None:
        from physiosight.capture.sources import open_source
        from physiosight.perception.mediapipe_backend import MediaPipePoseBackend

        self.opts = opts
        self.source = open_source(opts.source)
        self.backend = MediaPipePoseBackend(opts.pose_model)
        self.hands = None
        if opts.hand_model and Path(opts.hand_model).is_file():
            from physiosight.perception.hands import MediaPipeHandBackend

            self.hands = MediaPipeHandBackend(opts.hand_model)
        elif opts.hand_model:
            print(f"Hand model not found at {opts.hand_model}; hand tracking disabled.")
        self.notifier = TelegramNotifier()
        fps = self.source.nominal_fps or 30.0
        self.clips = ClipRecorder(opts.clips_dir, fps, cooldown_s=opts.clip_cooldown_s,
                                  on_saved=self._send_clip) if opts.clips_dir else None
        self.exercise, self.side = opts.exercise, opts.side
        self.sessions: list[Session] = []
        self._last_fault_wall = 0.0
        self._notified_complete = False
        self.analysis = self._new_analysis()

    # -------------------------------------------------------------- wiring --
    def _new_analysis(self) -> Analysis:
        spec, sha = load_spec(self.exercise, self.side)
        self.side = spec.sides[0]
        self.spec = spec
        target = self.opts.target or spec.default_target
        info = SourceInfo(kind="camera" if self.source.kind == "camera" else "video",
                          width=self.source.width or None, height=self.source.height or None,
                          nominal_fps=self.source.nominal_fps, mirrored=False)
        smoother = None
        if self.opts.smoothing:
            from physiosight.signal.filters import LandmarkSmoother

            smoother = LandmarkSmoother()
        a = Analysis(spec, target, source=info, pose_backend=self.backend.name,
                     pose_backend_version=self.backend.version, params=self.opts.params,
                     spec_sha256=sha, smoother=smoother)
        a.bus.subscribe(FaultDetected, self._on_fault)
        a.bus.subscribe(RepFinished, lambda ev: print(f"[{ev.t:7.2f}s] rep {ev.record.index}: {ev.record.outcome.value}"
                                                       + (f" {ev.record.faults}" if ev.record.faults else "")))
        a.bus.subscribe(TargetReached, self._on_complete)
        if self.clips:
            self.clips.attach(a.bus)
        self._notified_complete = False
        print(f"Now tracking: {i18n.text(spec.name_key, 'en')} ({self.side}), target {target:g}")
        return a

    def _end_segment(self) -> None:
        session = self.analysis.finish()
        if session.tracking.frames_total:
            self.sessions.append(session)
            if self.opts.sessions_dir:
                d = Path(self.opts.sessions_dir)
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{time.strftime('%Y%m%d_%H%M%S')}_{session.exercise_id}_{str(session.session_id)[:8]}.json"
                path.write_text(session.model_dump_json(indent=2) + "\n")
                print(f"Session saved: {path}")

    def _on_fault(self, ev: FaultDetected) -> None:
        self._last_fault_wall = time.time()
        print(f"[{ev.t:7.2f}s] fault: {ev.code} - {i18n.text('fault.' + ev.code, self.opts.lang)}")

    def _send_clip(self, path: Path, ev: FaultDetected) -> None:
        print(f"[ALERT] clip saved: {path}")
        self.notifier.send_video(path, caption=f"PhysioSight: {i18n.text(self.spec.name_key, self.opts.lang)} - "
                                               f"{i18n.text('fault.' + ev.code, self.opts.lang)}")

    def _on_complete(self, _ev: TargetReached) -> None:
        if not self._notified_complete:
            a = self.analysis.analyzer
            self.notifier.send_message(f"PhysioSight: {i18n.text(self.spec.name_key, self.opts.lang)} complete "
                                       f"{a.progress:g}/{a.target:g}")
            self._notified_complete = True

    # ---------------------------------------------------------------- loop --
    def run(self) -> list[Session]:
        import cv2

        from physiosight.render.overlay import draw_hud, draw_pose

        print("Telegram alerts:", "ENABLED" if self.notifier.enabled else
              "disabled (set PHYSIOSIGHT_TG_TOKEN and PHYSIOSIGHT_TG_CHAT)")
        self.notifier.send_message("PhysioSight: session started.")
        mirror = self.source.kind == "camera"
        try:
            for t, frame in self.source.frames():
                pose = self.backend.process(frame, t)
                hands = self.hands.process(frame, t) if self.hands else []
                result = self.analysis.step(t, pose)
                view = cv2.flip(frame, 1) if mirror else frame.copy()
                w = view.shape[1]
                shown_pose = mirror_pose(pose, w) if mirror else pose
                shown_hands = [np.c_[w - h[:, 0], h[:, 1]] for h in hands] if mirror else hands
                draw_pose(view, shown_pose, shown_hands, cover_face=self.opts.cover_face)
                if self.clips:
                    self.clips.add(view)  # the buffered frames are the anonymised view
                unit = "sec" if self.spec.kind == "hold" else "reps"
                draw_hud(view, result, i18n.text(self.spec.name_key, "en") + (
                    f" ({self.side})" if self.side != "bilateral" else ""), self.analysis.analyzer.target, unit,
                    fault_active=time.time() - self._last_fault_wall < 1.0,
                    complete=self.analysis.analyzer.is_complete)
                if self.opts.headless:
                    if self.analysis.done and self.source.kind != "camera":
                        break
                    continue
                cv2.imshow("PhysioSight (1-5 exercise, a side, r reset, q quit)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self._end_segment()
                    self.analysis = self._new_analysis()
                elif key == ord("a") and len(load_spec(self.exercise)[0].sides) >= 1 and self.side in ("left", "right"):
                    self._end_segment()
                    self.side = "left" if self.side == "right" else "right"
                    self.analysis = self._new_analysis()
                elif key in EXERCISE_KEYS:
                    self._end_segment()
                    self.exercise, self.side = EXERCISE_KEYS[key], None
                    self.analysis = self._new_analysis()
        finally:
            self._end_segment()
            self.notifier.flush()
            self.source.close()
            self.backend.close()
            if self.hands:
                self.hands.close()
            if not self.opts.headless:
                cv2.destroyAllWindows()
        print("\n--- Session summary ---")
        for s in self.sessions:
            unit = "s" if s.exercise_kind == "hold" else "reps"
            print(f"{i18n.text('exercise.' + s.exercise_id, 'en')} ({s.side}): {s.summary.achieved:g}/"
                  f"{s.summary.target:g} {unit}, complete: {'YES' if s.summary.completed else 'no'}")
        return self.sessions
