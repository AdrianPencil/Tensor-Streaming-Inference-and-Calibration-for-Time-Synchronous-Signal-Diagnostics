"""
ssp.streaming.replay

Deterministic replay/backfill runner.

This module enforces a consistent "arrival order" iteration and exposes
watermark updates so downstream components can decide when to finalize windows.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar

from ssp.core.types import Event
from ssp.streaming.event_time import EventTimePolicy, WatermarkState, advance_watermark

__all__ = ["ReplayRunner"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ReplayRunner:
    """Replay a stream of events deterministically while tracking watermark."""

    policy: EventTimePolicy

    def run(
        self,
        events: Iterable[Event],
        on_event: Callable[[Event, WatermarkState], T],
    ) -> Iterator[T]:
        state = WatermarkState()
        for ev in events:
            state = advance_watermark(state, float(ev.t_event), self.policy)
            yield on_event(ev, state)
