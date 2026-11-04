"""MediaPipe Pose Landmarker (BlazePose, 33 keypoints) backend.

Differences from the legacy ``pose_estimator.py``:

* timestamps come from the video (``t``) instead of ``time.monotonic()``, so
  offline results do not depend on processing speed;
* outputs pixel coordinates plus world landmarks (metres, hip-centred).
"""
from __future__ import annotations

import numpy as np

from physiosight.perception.base import BLAZEPOSE_33, PoseFrame


class MediaPipePoseBackend:
    skeleton = "blazepose33"

    def __init__(
        self,
        model_path: str,
        min_pose_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install the 'mediapipe' extra: pip install physiosight[mediapipe]") from exc
        self._mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1
        self.name = f"mediapipe:{model_path.rsplit('/', 1)[-1].removesuffix('.task')}"
        self.version = getattr(mp, "__version__", None)

    def process(self, frame_bgr: np.ndarray, t: float) -> PoseFrame | None:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(round(t * 1000)), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        k = len(BLAZEPOSE_33)
        xy = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float).reshape(k, 2)
        vis = np.array([float(lm.visibility or 0.0) for lm in lms])
        pres = np.array([float(lm.presence or 0.0) for lm in lms])
        world = None
        if result.pose_world_landmarks:
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=float
            )
        return PoseFrame(t, self.skeleton, xy, vis, pres, (w, h), world)

    def close(self) -> None:
        self._landmarker.close()
