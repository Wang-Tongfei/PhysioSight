"""Drawing: skeleton, hands, anonymous head, HUD panels. Pure presentation.

Analysis always runs on unflipped frames so left/right stay anatomical; the
renderer may mirror the *output* for a mirror-like live camera view.

Privacy: by default the head is covered by a filled oval (placed from ears and
shoulders only - eye/nose/mouth keypoints are not used), so recorded fault
clips no longer show the face. The legacy monitor drew only an outline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from physiosight import i18n
from physiosight.exercises.machines import FrameResult
from physiosight.perception.base import PoseFrame
from physiosight.perception.hands import HAND_BONES

BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30),
         (30, 32), (28, 32)]
BODY_POINTS = set(range(11, 17)) | set(range(23, 33))
GREEN, RED, WHITE, BLACK, YELLOW, CYAN = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (0, 215, 255), (255, 220, 0)


@dataclass(frozen=True)
class HeadOval:
    centre: tuple[int, int]
    axes: tuple[int, int]
    angle: float
    neck: tuple[int, int]


def head_oval(xy: np.ndarray) -> HeadOval | None:
    """Head oval from ears + shoulders (+ hips as fallback), BlazePose indices."""
    le, re, ls, rs, lh, rh = (xy[i] for i in (7, 8, 11, 12, 23, 24))
    if not np.all(np.isfinite([le, re, ls, rs, lh, rh])):
        return None
    sh_mid, hip_mid, ear_mid = (ls + rs) / 2, (lh + rh) / 2, (le + re) / 2
    sh_w, ear_w = float(np.linalg.norm(ls - rs)), float(np.linalg.norm(le - re))
    up = ear_mid - sh_mid
    if np.linalg.norm(up) < 1:
        up = sh_mid - hip_mid
    if np.linalg.norm(up) < 1:
        return None
    up = up / np.linalg.norm(up)
    scale = max(sh_w, float(np.linalg.norm(sh_mid - hip_mid)) * 0.45)
    if scale < 4:
        return None
    ears_ok = scale * 0.18 <= ear_w <= scale * 0.9
    face_w = ear_w if ears_ok else scale * 0.48
    if ears_ok:
        c = ear_mid + up * face_w * 0.12
        angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    else:
        c = sh_mid + up * scale * 0.72
        angle = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
    neck = sh_mid + up * scale * 0.18
    return HeadOval((int(c[0]), int(c[1])), (max(8, int(face_w * 0.62)), max(11, int(face_w * 0.82))),
                    angle, (int(neck[0]), int(neck[1])))


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, colour=BLACK, alpha: float = 0.55) -> None:
    import cv2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_pose(frame: np.ndarray, pose: PoseFrame | None, hands: list[np.ndarray] | None = None,
              cover_face: bool = True) -> None:
    import cv2

    if pose is not None and pose.skeleton == "blazepose33":
        ok = np.isfinite(pose.xy).all(axis=1) & (pose.visibility >= 0.5)
        pts = [tuple(int(v) for v in p) if good else None for p, good in zip(pose.xy, ok, strict=True)]
        oval = head_oval(pose.xy)
        if oval is not None:
            if cover_face:
                cv2.ellipse(frame, oval.centre, (int(oval.axes[0] * 1.25), int(oval.axes[1] * 1.25)),
                            oval.angle, 0, 360, (80, 80, 80), -1)
            cv2.ellipse(frame, oval.centre, oval.axes, oval.angle, 0, 360, YELLOW, 2)
            for s in (11, 12):
                ps = pts[s]
                if ps is not None:
                    cv2.line(frame, ps, oval.neck, YELLOW, 2)
        for a, b in BONES:
            pa, pb = pts[a], pts[b]
            if pa is not None and pb is not None:
                cv2.line(frame, pa, pb, YELLOW, 2)
        for i in BODY_POINTS:
            pi = pts[i]
            if pi is not None:
                cv2.circle(frame, pi, 4, WHITE, -1)
    for hand in hands or []:
        hp = [tuple(int(v) for v in p) for p in hand]
        for a, b in HAND_BONES:
            cv2.line(frame, hp[a], hp[b], CYAN, 2)
        for p in hp:
            cv2.circle(frame, p, 3, WHITE, -1)


def draw_hud(frame: np.ndarray, result: FrameResult | None, exercise_name: str, target: float,
             unit: str, fault_active: bool = False, complete: bool = False, lang: str = "en") -> None:
    """Panels and text. OpenCV's Hershey fonts cannot render CJK, so on-frame text
    is English; localised strings are available via ``i18n.text(key, lang)``."""
    import cv2

    del lang
    h, w = frame.shape[:2]
    _panel(frame, 0, 0, w, 90)
    _panel(frame, 0, h - 45, w, 45)
    progress = 0.0 if result is None else result.progress
    prog = f"{progress:.1f}/{target:.0f} SEC" if unit == "sec" else f"{int(progress)}/{int(target)} REPS"
    cv2.putText(frame, prog, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(frame, f"Exercise: {exercise_name}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    if result is not None:
        if result.metric_value is not None and result.metric_label_key:
            label = i18n.text(result.metric_label_key, "en")
            cv2.putText(frame, f"{label}: {int(result.metric_value)} deg", (w - 230, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        values = {"progress": int(progress) if unit != "sec" else progress, "target": target}
        msg = i18n.text(result.hint_key, "en", **values)
        cv2.putText(frame, f"Form: {msg}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    GREEN if result.form_ok else RED, 2)
    else:
        cv2.putText(frame, "Form: Step into view of the camera", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    if fault_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 8)
    if complete:
        _panel(frame, 0, h // 2 - 50, w, 100, colour=(0, 120, 0), alpha=0.7)
        cv2.putText(frame, "WORKOUT COMPLETE", (w // 2 - 220, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
