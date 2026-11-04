"""Frame sources yielding ``(t_seconds, frame_bgr)``.

Video files use container timestamps (falling back to index / fps), so the
analysis is reproducible regardless of how fast frames are processed. Cameras
use a monotonic clock relative to the first frame.

Mirroring is a *display* concern. Flipping before pose estimation swaps the
anatomical left/right labels (the legacy CLI did this for every source), so
sources here never flip; the renderer mirrors the drawn output instead.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterator


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the 'video' extra: pip install physiosight[video]") from exc
    return cv2


class VideoFileSource:
    kind = "video"

    def __init__(self, path: str) -> None:
        cv2 = _cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.nominal_fps = fps if 1.0 < fps < 240.0 else None
        self.label = f"file:{path}"

    def frames(self) -> Iterator[tuple[float, object]]:
        cv2 = self._cv2
        index = 0
        last_t = -1.0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if self.nominal_fps:
                t_index = index / self.nominal_fps
            else:
                t_index = index / 30.0
            t = pos_ms / 1000.0 if pos_ms > 0 and pos_ms / 1000.0 > last_t else t_index
            if t <= last_t:
                t = last_t + 1e-3
            last_t = t
            index += 1
            yield t, frame

    def close(self) -> None:
        self._cap.release()


class CameraSource:
    """Webcam. Tries several OpenCV backends (DirectShow and Media Foundation on
    Windows) and rejects devices that only deliver black frames, then scans the
    next indices - the same fallback the legacy monitor used."""

    kind = "camera"

    def __init__(self, index: int = 0, scan: int = 3) -> None:
        cv2 = _cv2()
        cap, backend, used = None, "", index
        for candidate in [index] + [i for i in range(scan) if i != index]:
            cap, backend = _open_live_camera(cv2, candidate)
            if cap is not None:
                used = candidate
                break
        if cap is None:
            raise RuntimeError("no working camera detected")
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.nominal_fps = fps if 1.0 < fps < 240.0 else None
        self.label = f"camera:{used} ({backend})"

    def frames(self) -> Iterator[tuple[float, object]]:
        t0 = None
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            now = time.monotonic()
            t0 = now if t0 is None else t0
            yield now - t0, frame

    def close(self) -> None:
        self._cap.release()


def _open_live_camera(cv2, index: int):
    backends = [("default", cv2.CAP_ANY)]
    if sys.platform == "win32":
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)] + backends
    for name, api in backends:
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(15):  # sensors may need a few frames to warm up
            ok, frame = cap.read()
            if ok and frame is not None and float(frame.mean()) > 10:
                return cap, name
            time.sleep(0.05)
        cap.release()
    return None, None


def open_source(spec: str):
    """``"0"`` -> camera 0, anything else -> video file path."""
    if spec.isdigit():
        return CameraSource(int(spec))
    return VideoFileSource(spec)
