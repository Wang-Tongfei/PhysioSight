"""The synthetic baseline suite: deterministic landmark sequences + what we expect.

``expect="identical"``: the new engine must reproduce the legacy engine frame
by frame (count after every frame and every fault event).
``expect="BC-n"``: an intentional behaviour change documented in
docs/BEHAVIOR_CHANGES.md; the test then checks the *new* result against the
ground truth of the synthetic script and that legacy differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physiosight.perception.cache import LandmarkSequence
from physiosight.testing import synth

SCRIPTS = {
    "squat": synth.squat_script, "bicep_curl": synth.curl_script, "pushup": synth.pushup_script,
    "plank": synth.plank_script,
}


@dataclass(frozen=True)
class Case:
    name: str
    exercise: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    fps: float = 30.0
    size: tuple[int, int] = (720, 720)
    noise_px: float = 0.0
    seed: int = 7
    target: float = 10
    expect: str = "identical"

    def script(self) -> synth.Script:
        return SCRIPTS[self.exercise](**self.kwargs)

    def sequence(self) -> LandmarkSequence:
        return synth.render(self.script(), fps=self.fps, size=self.size, noise_px=self.noise_px, seed=self.seed)


def _rep_cases() -> list[Case]:
    out: list[Case] = []
    variants = {
        "squat": [("clean", {}), ("shallow", {"depth": 120}), ("uneven", {"asym": 30}),
                  ("too_fast", {"down_s": 0.2, "up_s": 0.2})],
        "bicep_curl": [("clean", {}), ("partial", {"top": 70}), ("drift", {"drift": 50}),
                       ("too_fast", {"up_s": 0.08, "down_s": 0.08})],
        "pushup": [("clean", {}), ("shallow", {"bottom": 120}), ("too_fast", {"down_s": 0.2, "up_s": 0.2}),
                   ("sagging", {"sag": 60})],
    }
    for ex, items in variants.items():
        for name, kw in items:
            for noise in (0.0, 1.5):
                out.append(Case(f"{ex}-{name}-noise{noise:g}", ex, {"reps": 3, **kw}, noise_px=noise))
    return out


CASES: list[Case] = [
    *_rep_cases(),
    Case("plank-good", "plank", {"hold_s": 6.0}, target=30),
    Case("plank-sag25", "plank", {"hold_s": 6.0, "sag": 25.0}, target=30),
    Case("plank-sag60", "plank", {"hold_s": 6.0, "sag": 60.0}, target=30),
    Case("squat-target-stop", "squat", {"reps": 5}, target=2),
    # intentional behaviour changes
    Case("BC1-squat-16x9", "squat", {"reps": 3, "depth": 95}, size=(1280, 720), expect="BC-1"),
    Case("BC1-pushup-16x9", "pushup", {"reps": 3}, size=(1280, 720), expect="BC-1"),
    Case("BC2-squat-10fps", "squat", {"reps": 3, "depth": 88, "bottom_s": 0.1, "pause_s": 0.2, "down_s": 0.7,
                                      "up_s": 0.7}, fps=10, expect="BC-2"),
]
