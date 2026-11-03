"""Synthetic BlazePose-33 landmark sequences with known ground truth.

A 2D sagittal-plane stick figure is posed by forward kinematics from a few
joint angles, which are interpolated between keyframes. This gives exact
ground-truth angles and repetition counts for testing the analysis layer
(geometry, state machines, frame-rate invariance). It says nothing about
pose-model accuracy - that needs real video and a reference system.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from physiosight.perception.base import BLAZEPOSE_33
from physiosight.perception.cache import LandmarkSequence

IDX = {n: i for i, n in enumerate(BLAZEPOSE_33)}
Params = dict[str, float]


def _rot(v: tuple[float, float], deg: float) -> np.ndarray:
    """Rotate a 2D vector counter-clockwise on screen (y down) by deg."""
    r = math.radians(deg)
    x, y = v
    return np.array([x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)])


def _unit(deg_from_down: float) -> np.ndarray:
    """Unit vector pointing down rotated by deg toward +x (screen right)."""
    r = math.radians(deg_from_down)
    return np.array([math.sin(r), math.cos(r)])


@dataclass
class Body:
    scale: float = 300.0  # pixels for a 1.0-unit segment
    shank: float = 0.45
    thigh: float = 0.45
    torso: float = 0.55
    upper_arm: float = 0.32
    forearm: float = 0.28
    depth_offset: float = 6.0  # px shift between near and far side


def _fill_extremities(p: dict[str, np.ndarray], facing: float) -> None:
    """Place face, hands and feet relative to the main joints."""
    sh = (p["left_shoulder"] + p["right_shoulder"]) / 2
    hip = (p["left_hip"] + p["right_hip"]) / 2
    up = sh - hip
    n = np.linalg.norm(up) or 1.0
    up = up / n
    head = sh + up * 60 + np.array([facing * 10, 0])
    for name in ("nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
                 "right_eye", "right_eye_outer", "mouth_left", "mouth_right"):
        p[name] = head + np.array([facing * 12, 0])
    p["left_ear"] = head + np.array([-3, 0])
    p["right_ear"] = head + np.array([3, 0])
    for side in ("left", "right"):
        w = p[f"{side}_wrist"]
        e = p[f"{side}_elbow"]
        d = w - e
        d = d / (np.linalg.norm(d) or 1.0)
        for name in ("pinky", "index", "thumb"):
            p[f"{side}_{name}"] = w + d * 15
        a = p[f"{side}_ankle"]
        p[f"{side}_heel"] = a + np.array([-facing * 12, 8])
        p[f"{side}_foot_index"] = a + np.array([facing * 30, 10])


def squat_pose(q: Params, body: Body, origin: np.ndarray) -> dict[str, np.ndarray]:
    """Side view, facing +x. q: knee_l, knee_r (interior deg)."""
    p: dict[str, np.ndarray] = {}
    s = body.scale
    for side, sign in (("left", -1), ("right", 1)):
        k = q[f"knee_{side[0]}"]
        flex = 180.0 - k
        phi = 0.45 * flex  # shank leans forward
        theta = flex - phi  # thigh leans back
        ankle = origin + np.array([sign * body.depth_offset, 0.0])
        knee = ankle + s * body.shank * np.array([math.sin(math.radians(phi)), -math.cos(math.radians(phi))])
        hip = knee + s * body.thigh * np.array([-math.sin(math.radians(theta)), -math.cos(math.radians(theta))])
        p[f"{side}_ankle"], p[f"{side}_knee"], p[f"{side}_hip"] = ankle, knee, hip
    hip_mid = (p["left_hip"] + p["right_hip"]) / 2
    lean = 0.35 * (180.0 - (q["knee_l"] + q["knee_r"]) / 2)
    for side, sign in (("left", -1), ("right", 1)):
        sh = hip_mid + np.array([sign * body.depth_offset, 0]) + s * body.torso * np.array(
            [math.sin(math.radians(lean)), -math.cos(math.radians(lean))])
        elbow = sh + s * body.upper_arm * np.array([0.9, 0.1])  # arms reach forward
        wrist = elbow + s * body.forearm * np.array([1.0, 0.0])
        p[f"{side}_shoulder"], p[f"{side}_elbow"], p[f"{side}_wrist"] = sh, elbow, wrist
    _fill_extremities(p, 1.0)
    return p


def curl_pose(q: Params, body: Body, origin: np.ndarray) -> dict[str, np.ndarray]:
    """Side view standing, facing +x. q: elbow (interior), drift (upper arm vs torso)."""
    p: dict[str, np.ndarray] = {}
    s = body.scale
    for side, sign in (("left", -1), ("right", 1)):
        ankle = origin + np.array([sign * body.depth_offset, 0.0])
        knee = ankle + np.array([0, -s * body.shank])
        hip = knee + np.array([0, -s * body.thigh])
        sh = hip + np.array([0, -s * body.torso])
        drift = q.get("drift", 0.0) if side == "right" else q.get("drift_l", 0.0)
        elbow_angle = q["elbow"] if side == "right" else q.get("elbow_l", 170.0)
        elbow = sh + s * body.upper_arm * _unit(drift)
        # forearm: start along the upper arm direction (straight), fold forward
        fore_dir = _unit(drift + (180.0 - elbow_angle))
        wrist = elbow + s * body.forearm * fore_dir
        p.update({f"{side}_ankle": ankle, f"{side}_knee": knee, f"{side}_hip": hip,
                  f"{side}_shoulder": sh, f"{side}_elbow": elbow, f"{side}_wrist": wrist})
    _fill_extremities(p, 1.0)
    return p


def _prone(q: Params, body: Body, origin: np.ndarray, forearm_plank: bool) -> dict[str, np.ndarray]:
    """Side view, head toward +x. q: elbow (interior, push-up), sag (hip drop px)."""
    p: dict[str, np.ndarray] = {}
    s = body.scale
    lu, lf = s * body.upper_arm, s * body.forearm
    if forearm_plank:
        h = lu
    else:
        e = math.radians(q["elbow"])
        h = math.sqrt(lu * lu + lf * lf - 2 * lu * lf * math.cos(e))
    body_len = s * (body.torso + body.thigh + body.shank)
    for side, sign in (("left", -1), ("right", 1)):
        off = np.array([sign * body.depth_offset, 0.0])
        wrist = origin + off
        sh = wrist + np.array([0.0, -h])
        if forearm_plank:
            elbow = sh + np.array([0.0, lu])
            wrist = elbow + np.array([lf, 0.0])
        else:
            # elbow = intersection of circles around wrist (lf) and shoulder (lu), toward feet
            d = h
            a = (lf * lf - lu * lu + d * d) / (2 * d)
            hh = math.sqrt(max(lf * lf - a * a, 0.0))
            base = wrist + np.array([0.0, -a])
            elbow = base + np.array([-hh, 0.0])
        dx = math.sqrt(max(body_len ** 2 - h ** 2, 1.0))
        ankle = sh + np.array([-dx, h])
        frac = body.torso / (body.torso + body.thigh + body.shank)
        hip = sh + frac * (ankle - sh) + np.array([0.0, q.get("sag", 0.0)])
        knee = hip + 0.5 * (ankle - hip)
        p.update({f"{side}_ankle": ankle, f"{side}_knee": knee, f"{side}_hip": hip,
                  f"{side}_shoulder": sh, f"{side}_elbow": elbow, f"{side}_wrist": wrist})
    _fill_extremities(p, 1.0)
    return p


def pushup_pose(q: Params, body: Body, origin: np.ndarray) -> dict[str, np.ndarray]:
    return _prone(q, body, origin, forearm_plank=False)


def plank_pose(q: Params, body: Body, origin: np.ndarray) -> dict[str, np.ndarray]:
    return _prone(q, body, origin, forearm_plank=True)


POSERS: dict[str, Callable[[Params, Body, np.ndarray], dict[str, np.ndarray]]] = {
    "squat": squat_pose, "bicep_curl": curl_pose, "pushup": pushup_pose, "plank": plank_pose,
}


@dataclass
class Script:
    """Piecewise-linear keyframes: list of (duration_s, target params)."""

    exercise: str
    start: Params
    moves: list[tuple[float, Params]] = field(default_factory=list)
    expected_valid: int = 0  # ground truth number of valid repetitions
    expected_faults: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(d for d, _ in self.moves)

    def params_at(self, t: float) -> Params:
        cur, t0 = dict(self.start), 0.0
        for dur, target in self.moves:
            if t <= t0 + dur:
                a = (t - t0) / dur if dur > 0 else 1.0
                return {k: cur[k] + a * (target.get(k, cur[k]) - cur[k]) for k in cur}
            cur = {**cur, **target}
            t0 += dur
        return cur


def render(
    script: Script,
    fps: float = 30.0,
    size: tuple[int, int] = (1280, 720),
    noise_px: float = 0.0,
    seed: int = 0,
    dropout: Callable[[float], bool] | None = None,
    body: Body | None = None,
) -> LandmarkSequence:
    """Sample a script into a landmark sequence."""
    body = body or Body(scale=min(size) / 2.2)
    rng = np.random.default_rng(seed)
    w, h = size
    origin = {
        "squat": np.array([w * 0.45, h * 0.92]),
        "bicep_curl": np.array([w * 0.45, h * 0.95]),
        "pushup": np.array([w * 0.70, h * 0.90]),
        "plank": np.array([w * 0.70, h * 0.90]),
    }[script.exercise]
    poser = POSERS[script.exercise]
    n = int(math.floor(script.duration * fps)) + 1
    t = np.arange(n) / fps
    k = len(BLAZEPOSE_33)
    xy = np.zeros((n, k, 2))
    vis = np.ones((n, k))
    for i, ti in enumerate(t):
        pts = poser(script.params_at(float(ti)), body, origin)
        for name, v in pts.items():
            xy[i, IDX[name]] = v
        if noise_px:
            xy[i] += rng.normal(0, noise_px, size=(k, 2))
        if dropout is not None and dropout(float(ti)):
            vis[i] = 0.1
    return LandmarkSequence("blazepose33", w, h, t, np.ones(n, bool), xy, vis, vis.copy(),
                            meta={"synthetic": True, "exercise": script.exercise, "fps": fps})


# ------------------------------------------------------------ scripts ----
def squat_script(reps: int = 3, depth: float = 90.0, down_s: float = 1.0, up_s: float = 1.0,
                 asym: float = 0.0, pause_s: float = 0.6, bottom_s: float = 0.3) -> Script:
    stand = {"knee_l": 176.0, "knee_r": 176.0}
    moves: list[tuple[float, Params]] = [(1.0, stand)]
    for _ in range(reps):
        moves += [(down_s, {"knee_l": depth, "knee_r": depth + asym}), (bottom_s, {}),
                  (up_s, stand), (pause_s, {})]
    ok = depth <= 98 and asym <= 18 and down_s + up_s + bottom_s >= 0.9
    return Script("squat", stand, moves, expected_valid=reps if ok else 0)


def curl_script(reps: int = 3, top: float = 35.0, up_s: float = 1.0, down_s: float = 1.0,
                drift: float = 5.0, pause_s: float = 0.6) -> Script:
    rest = {"elbow": 175.0, "drift": drift}
    moves: list[tuple[float, Params]] = [(1.0, rest)]
    for _ in range(reps):
        moves += [(up_s, {"elbow": top}), (0.3, {}), (down_s, {"elbow": 175.0}), (pause_s, {})]
    ok = top <= 43 and drift <= 33 and up_s + down_s + 0.3 >= 0.7
    return Script("bicep_curl", rest, moves, expected_valid=reps if ok else 0)


def pushup_script(reps: int = 3, bottom: float = 85.0, down_s: float = 1.0, up_s: float = 1.0,
                  sag: float = 0.0, pause_s: float = 0.6) -> Script:
    top = {"elbow": 172.0, "sag": sag}
    moves: list[tuple[float, Params]] = [(1.0, top)]
    for _ in range(reps):
        moves += [(down_s, {"elbow": bottom}), (0.3, {}), (up_s, {"elbow": 172.0}), (pause_s, {})]
    return Script("pushup", top, moves, expected_valid=reps if bottom <= 98 else 0)


def plank_script(hold_s: float = 10.0, sag: float = 0.0) -> Script:
    q = {"sag": sag}
    return Script("plank", q, [(hold_s, q)])
