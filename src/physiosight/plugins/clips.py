"""Save the last few seconds of (already anonymised) frames when a fault fires."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from physiosight.events import EventBus, FaultDetected


class ClipRecorder:
    def __init__(self, out_dir: str | Path, fps: float, buffer_s: float = 4.0,
                 cooldown_s: float = 15.0, on_saved: Callable[[Path, FaultDetected], None] | None = None):
        self.out_dir = Path(out_dir)
        self.fps = fps
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, int(buffer_s * fps)))
        self._cooldown = cooldown_s
        self._last: dict[str, float] = {}
        self._on_saved = on_saved
        self.saved: list[Path] = []

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(FaultDetected, self._on_fault)

    def add(self, frame: np.ndarray) -> None:
        self._buf.append(frame.copy())

    def _on_fault(self, ev: FaultDetected) -> None:
        if ev.t - self._last.get(ev.code, -1e9) < self._cooldown or not self._buf:
            return
        self._last[ev.code] = ev.t
        import cv2

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{ev.code}_{time.strftime('%Y%m%d_%H%M%S')}_{ev.t:.1f}.mp4"
        h, w = self._buf[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        for frame in self._buf:
            writer.write(frame)
        writer.release()
        self.saved.append(path)
        if self._on_saved:
            self._on_saved(path, ev)
