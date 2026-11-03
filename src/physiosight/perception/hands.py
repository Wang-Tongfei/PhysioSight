"""MediaPipe Hand Landmarker (21 keypoints per hand), used for display.

Hand keypoints are not used by any exercise spec yet; they are drawn so that
therapists can see grip/finger position, as in the legacy monitor.
"""
from __future__ import annotations

import numpy as np

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class MediaPipeHandBackend:
    def __init__(self, model_path: str, num_hands: int = 2) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> list[np.ndarray]:
        """Return 0-2 arrays of shape (21, 2) in pixels."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(round(t * 1000)), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)
        return [np.array([[lm.x * w, lm.y * h] for lm in hand]) for hand in result.hand_landmarks]

    def close(self) -> None:
        self._landmarker.close()
