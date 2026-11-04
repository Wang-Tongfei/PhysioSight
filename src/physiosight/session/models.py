"""Session data contract shared by the engine, the platform and the research repo.

This module only depends on Pydantic so that consumers (e.g. the platform
backend) can install the engine without computer-vision dependencies.

Conventions
-----------
* Times are seconds relative to the first analysed frame (``t=0``).
* Angles are degrees. Their meaning is defined by the exercise spec that
  produced them (see ``RepRecord.angles``); names follow
  ``<joint>_<motion>[_<side>]``, e.g. ``knee_flexion_left`` where 0 deg means
  full extension (clinical goniometry convention).
* ``angle_space`` says whether angles were measured in the image plane
  (``image_2d``) or from 3D landmarks (``world_3d``). 2D angles are projections
  and depend on camera viewpoint.
* Fault codes are snake_case identifiers declared by exercise specs. Human
  readable text lives in locale files, never in the Session.
* A Session carries no patient identity and no images. The platform
  associates it with a patient on upload.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["0.1.0"] = "0.1.0"

FaultCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
AngleName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExerciseKind(StrEnum):
    REPETITION = "repetition"
    HOLD = "hold"


class AngleSpace(StrEnum):
    IMAGE_2D = "image_2d"
    WORLD_3D = "world_3d"


class SourceInfo(_Model):
    kind: Literal["camera", "video", "landmarks", "synthetic"]
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    nominal_fps: float | None = Field(default=None, gt=0)
    effective_fps: float | None = Field(
        default=None, gt=0, description="Analysed frames / duration."
    )
    mirrored: bool = Field(
        default=False,
        description="True if frames were horizontally flipped before pose "
        "estimation; left/right landmark labels are then swapped anatomically.",
    )


class EngineInfo(_Model):
    name: str
    version: str
    pose_backend: str = Field(description="e.g. 'mediapipe:pose_landmarker_lite'")
    pose_backend_version: str | None = None
    angle_space: AngleSpace = AngleSpace.IMAGE_2D
    smoothing: str | None = Field(default=None, description="e.g. 'one_euro(1.0,0.01)'")
    spec_id: str
    spec_version: str
    spec_sha256: str | None = None
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Fully resolved thresholds used for this session "
        "(spec defaults -> patient calibration -> session overrides).",
    )


class TrackingQuality(_Model):
    frames_total: int = Field(ge=0)
    frames_with_pose: int = Field(ge=0)
    frames_reliable: int = Field(
        ge=0, description="Frames where all required landmarks passed confidence gating."
    )
    tracking_lost_events: int = Field(default=0, ge=0)
    frames_view_ok: int | None = Field(
        default=None, ge=0,
        description="Reliable frames whose camera viewpoint matched the spec's view "
        "requirement (None if the spec has no view check).",
    )

    @property
    def reliable_ratio(self) -> float:
        return self.frames_reliable / self.frames_total if self.frames_total else 0.0

    @model_validator(mode="after")
    def _ordering(self) -> TrackingQuality:
        if not (self.frames_reliable <= self.frames_with_pose <= self.frames_total):
            raise ValueError("expected frames_reliable <= frames_with_pose <= frames_total")
        return self


class AngleExtrema(_Model):
    min_deg: float
    max_deg: float

    @property
    def range_deg(self) -> float:
        return self.max_deg - self.min_deg

    @model_validator(mode="after")
    def _ordering(self) -> AngleExtrema:
        if self.min_deg > self.max_deg:
            raise ValueError("min_deg must be <= max_deg")
        return self


class RepOutcome(StrEnum):
    VALID = "valid"
    INVALID = "invalid"  # completed the movement but failed a check
    ABORTED = "aborted"  # cancelled, e.g. tracking lost mid-repetition


class RepRecord(_Model):
    index: int = Field(ge=1, description="1-based index over all attempts.")
    started_at_s: float = Field(ge=0)
    ended_at_s: float = Field(ge=0)
    outcome: RepOutcome
    reached_target: bool
    faults: list[FaultCode] = Field(default_factory=list)
    angles: dict[AngleName, AngleExtrema] = Field(default_factory=dict)
    view_ok: bool | None = Field(
        default=None,
        description="True if >= 80% of the repetition's frames met the view requirement. "
        "2D angles from a mismatched viewpoint are projections and should not be "
        "interpreted as joint range of motion.",
    )

    @property
    def duration_s(self) -> float:
        return self.ended_at_s - self.started_at_s

    @model_validator(mode="after")
    def _consistency(self) -> RepRecord:
        if self.ended_at_s < self.started_at_s:
            raise ValueError("ended_at_s must be >= started_at_s")
        if self.outcome is RepOutcome.VALID and self.faults:
            raise ValueError("a valid repetition cannot carry faults")
        return self


class HoldSegment(_Model):
    started_at_s: float = Field(ge=0)
    ended_at_s: float = Field(ge=0)

    @property
    def duration_s(self) -> float:
        return self.ended_at_s - self.started_at_s


class FaultEvent(_Model):
    t_s: float = Field(ge=0)
    code: FaultCode
    rep_index: int | None = Field(default=None, ge=1)


class SessionSummary(_Model):
    target: float = Field(gt=0, description="Reps or seconds, depending on kind.")
    achieved: float = Field(ge=0, description="Valid reps, or credited hold seconds.")
    attempts: int = Field(default=0, ge=0)
    completed: bool
    duration_s: float = Field(ge=0)


class Session(_Model):
    schema_version: Literal["0.1.0"] = SCHEMA_VERSION
    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    exercise_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    exercise_kind: ExerciseKind
    side: Literal["left", "right", "bilateral"] = "bilateral"
    source: SourceInfo
    engine: EngineInfo
    tracking: TrackingQuality
    summary: SessionSummary
    reps: list[RepRecord] = Field(default_factory=list)
    holds: list[HoldSegment] = Field(default_factory=list)
    faults: list[FaultEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> Session:
        if self.exercise_kind is ExerciseKind.REPETITION:
            if self.holds:
                raise ValueError("repetition sessions carry reps, not holds")
            valid = sum(r.outcome is RepOutcome.VALID for r in self.reps)
            if valid != int(self.summary.achieved):
                raise ValueError("summary.achieved must equal the number of valid reps")
            if self.summary.attempts != len(self.reps):
                raise ValueError("summary.attempts must equal len(reps)")
            indices = [r.index for r in self.reps]
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError("rep indices must be 1..n in order")
        elif self.reps:
            raise ValueError("hold sessions carry holds, not reps")
        return self
