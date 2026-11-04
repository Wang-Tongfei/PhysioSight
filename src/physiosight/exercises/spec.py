"""Exercise specifications (YAML) and threshold resolution.

A spec declares *what* to measure (features), *when* phases are reached
(conditions written in :mod:`physiosight.exercises.expr`), which checks raise
which fault codes, and which angles to record per repetition. Two generic
state machines (repetition, hold) execute any spec; no exercise needs code.

Thresholds are layered: spec defaults -> patient calibration -> session
overrides, each validated against the bounds declared in the spec.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from physiosight.exercises.expr import Expr
from physiosight.kinematics.features import FeatureDef

Number = Union[float, str]  # literal or expression over params ("p.x")
STATE_VARS = {"state.ready", "state.in_movement", "state.at_target", "state.tracking_ok"}


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamDef(_M):
    default: float
    min: float = -math.inf
    max: float = math.inf
    unit: str = ""
    doc: str = ""

    @model_validator(mode="after")
    def _bounds(self) -> ParamDef:
        if not self.min <= self.default <= self.max:
            raise ValueError(f"default {self.default} outside [{self.min}, {self.max}]")
        return self


class Phase(_M):
    when: str
    hold_s: Number = 0.0


class Check(_M):
    """A condition that signals a fault while it is true (edge-triggered)."""

    id: str
    when: str
    fault: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope: Literal["always", "movement"] = "movement"
    invalidates_rep: bool = True


class HintRule(_M):
    when: str
    key: str


class Metric(_M):
    label_key: str
    expr: str


class Confidence(_M):
    min_visibility: float = Field(default=0.55, ge=0, le=1)
    min_presence: float = Field(default=0.50, ge=0, le=1)


class _SpecBase(_M):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str
    name_key: str
    skeleton: str = "blazepose33"
    sides: list[Literal["left", "right", "bilateral"]] = ["bilateral"]
    view: Literal["front", "side", "any"] = "any"
    default_target: float = Field(default=10, gt=0, description="reps, or seconds for holds")
    params: dict[str, ParamDef] = Field(default_factory=dict)
    confidence: Confidence = Confidence()
    required: list[str]
    features: dict[str, FeatureDef]
    derived: dict[str, str] = Field(default_factory=dict)
    checks: list[Check] = Field(default_factory=list)
    hints: list[HintRule] = Field(default_factory=list)
    default_hint: str
    tracking_hint: str
    primary_metric: Metric | None = None
    tracking_grace_s: Number = 0.0
    view_check: str | None = None

    @field_validator("params")
    @classmethod
    def _param_names(cls, v: dict[str, ParamDef]) -> dict[str, ParamDef]:
        for name in v:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"bad param name {name!r}")
        return v

    # -- validation of every expression against the declared names ---------
    def _expressions(self) -> list[tuple[str, str, set[str]]]:
        """(where, source, extra allowed names)."""
        out: list[tuple[str, str, set[str]]] = [(f"derived.{k}", v, set()) for k, v in self.derived.items()]
        out += [(f"checks.{c.id}", c.when, set()) for c in self.checks]
        out += [(f"hints[{i}]", h.when, STATE_VARS) for i, h in enumerate(self.hints)]
        if self.primary_metric:
            out.append(("primary_metric", self.primary_metric.expr, set()))
        if self.view_check:
            out.append(("view_check", self.view_check, set()))
        return out

    def _numbers(self) -> list[tuple[str, Number]]:
        return [("tracking_grace_s", self.tracking_grace_s)]

    @model_validator(mode="after")
    def _validate_names(self) -> _SpecBase:
        params = {f"p.{k}" for k in self.params}
        allowed = set(self.features) | params
        # derived may only use features, params and *earlier* derived names
        for name, src in self.derived.items():
            unknown = Expr.parse(src).names() - allowed
            if unknown:
                raise ValueError(f"derived.{name}: unknown names {sorted(unknown)}")
            allowed.add(name)
        for where, src, extra in self._expressions():
            unknown = Expr.parse(src).names() - allowed - extra
            if unknown:
                raise ValueError(f"{where}: unknown names {sorted(unknown)}")
        for where, num in self._numbers():
            if isinstance(num, str):
                unknown = Expr.parse(num).names() - params
                if unknown:
                    raise ValueError(f"{where}: may only reference params, got {sorted(unknown)}")
        if len({c.id for c in self.checks}) != len(self.checks):
            raise ValueError("check ids must be unique")
        return self


class RepetitionSpec(_SpecBase):
    kind: Literal["repetition"]
    ready: Phase
    start: Phase
    target: Phase
    finish: Phase = Field(alias="return")
    min_rep_duration_s: Number = 0.0
    too_fast_fault: str = "movement_too_fast"
    missed_target_fault: str
    record: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def _expressions(self) -> list[tuple[str, str, set[str]]]:
        out = super()._expressions()
        for name in ("ready", "start", "target", "finish"):
            out.append((name, getattr(self, name).when, set()))
        out += [(f"record.{k}", v, set()) for k, v in self.record.items()]
        return out

    def _numbers(self) -> list[tuple[str, Number]]:
        out = super()._numbers() + [("min_rep_duration_s", self.min_rep_duration_s)]
        return out + [(f"{n}.hold_s", getattr(self, n).hold_s)
                      for n in ("ready", "start", "target", "finish")]


class HoldSpec(_SpecBase):
    kind: Literal["hold"]
    valid_when: str
    max_dt_s: Number = 0.25
    reset_after_break_s: Number | None = None
    min_segment_s: Number = 0.0

    def _expressions(self) -> list[tuple[str, str, set[str]]]:
        return super()._expressions() + [("valid_when", self.valid_when, set())]

    def _numbers(self) -> list[tuple[str, Number]]:
        out = super()._numbers() + [("max_dt_s", self.max_dt_s), ("min_segment_s", self.min_segment_s)]
        if self.reset_after_break_s is not None:
            out.append(("reset_after_break_s", self.reset_after_break_s))
        return out


ExerciseSpec = Union[RepetitionSpec, HoldSpec]


# ---------------------------------------------------------------- loading --
def _substitute_side(obj: Any, side: str) -> Any:
    if isinstance(obj, str):
        return obj.replace("{side}", side)
    if isinstance(obj, list):
        return [_substitute_side(x, side) for x in obj]
    if isinstance(obj, dict):
        return {_substitute_side(k, side): _substitute_side(v, side) for k, v in obj.items()}
    return obj


def _builtin_dir() -> Path:
    return Path(str(resources.files("physiosight.exercises") / "specs"))


def list_builtin_specs() -> list[str]:
    return sorted(p.stem for p in _builtin_dir().glob("*.yaml"))


def load_spec(source: str | Path, side: str | None = None) -> tuple[ExerciseSpec, str]:
    """Load a spec by builtin id or file path. Returns (spec, sha256 of file)."""
    path = Path(source)
    if not path.suffix:
        path = _builtin_dir() / f"{source}.yaml"
    raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)
    sides = raw.get("sides", ["bilateral"])
    chosen = side or sides[0]
    if chosen not in sides:
        raise ValueError(f"{raw.get('id')}: side {chosen!r} not in {sides}")
    raw = _substitute_side(raw, chosen)
    raw["sides"] = [chosen]
    kind = raw.get("kind")
    model: type[RepetitionSpec] | type[HoldSpec]
    model = RepetitionSpec if kind == "repetition" else HoldSpec
    spec = model.model_validate(raw)
    return spec, hashlib.sha256(raw_text.encode()).hexdigest()


def resolve_params(spec: ExerciseSpec, *layers: Mapping[str, float] | None) -> dict[str, float]:
    """spec defaults <- calibration <- session overrides, bounds-checked."""
    out = {k: d.default for k, d in spec.params.items()}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key not in spec.params:
                raise KeyError(f"{spec.id}: unknown parameter {key!r}")
            d = spec.params[key]
            value = float(value)
            if not d.min <= value <= d.max:
                raise ValueError(f"{spec.id}.{key}={value} outside [{d.min}, {d.max}]")
            out[key] = value
    return out


def eval_number(value: Number | None, params: Mapping[str, float]) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return Expr.parse(value).evaluate({f"p.{k}": v for k, v in params.items()})
