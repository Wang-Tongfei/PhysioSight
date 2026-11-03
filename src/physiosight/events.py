"""Structured events emitted by the analysers.

Plugins (clip recording, notifications, UI) subscribe to events instead of
being called from inside the analysis loop.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, Union

from physiosight.session.models import HoldSegment, RepRecord


@dataclass(frozen=True, slots=True)
class FaultDetected:
    t: float
    code: str
    rep_index: int | None = None


@dataclass(frozen=True, slots=True)
class RepStarted:
    t: float
    index: int


@dataclass(frozen=True, slots=True)
class RepFinished:
    t: float
    record: RepRecord


@dataclass(frozen=True, slots=True)
class HoldSegmentClosed:
    t: float
    segment: HoldSegment


@dataclass(frozen=True, slots=True)
class TrackingLost:
    t: float


@dataclass(frozen=True, slots=True)
class TrackingRestored:
    t: float


@dataclass(frozen=True, slots=True)
class TargetReached:
    t: float


Event = Union[
    FaultDetected, RepStarted, RepFinished, HoldSegmentClosed, TrackingLost,
    TrackingRestored, TargetReached,
]
E = TypeVar("E")


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[type, list[Callable[[object], None]]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        self._subs[event_type].append(handler)  # type: ignore[arg-type]

    def publish(self, event: object) -> None:
        for handler in self._subs.get(type(event), ()):
            handler(event)
