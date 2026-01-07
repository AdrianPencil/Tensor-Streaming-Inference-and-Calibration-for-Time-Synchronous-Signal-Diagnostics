"""
ssp.streaming.event_time

Event-time primitives: timestamps, watermarks, and lateness policy.

The platform is event-time-first:
- processing order may differ from event-time order
- watermark is the platform's statement: "events earlier than this are too late"
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EventTimePolicy",
    "WatermarkState",
    "advance_watermark",
    "is_late",
]


@dataclass(frozen=True, slots=True)
class EventTimePolicy:
    """Defines allowed lateness and watermark lag in seconds."""

    allowed_lateness_s: float
    watermark_lag_s: float


@dataclass(frozen=True, slots=True)
class WatermarkState:
    """Mutable-in-practice watermark tracking state."""

    max_event_time_s: float = float("-inf")
    watermark_s: float = float("-inf")


def advance_watermark(state: WatermarkState, t_event_s: float, policy: EventTimePolicy) -> WatermarkState:
    """Advance watermark given an arriving event-time timestamp."""
    max_t = max(state.max_event_time_s, t_event_s)
    wm = max(state.watermark_s, max_t - policy.watermark_lag_s)
    return WatermarkState(max_event_time_s=max_t, watermark_s=wm)


def is_late(t_event_s: float | NDArray[np.float64], watermark_s: float, policy: EventTimePolicy) -> NDArray[np.bool_]:
    """Vectorized check for lateness under policy."""
    t = np.asarray(t_event_s, dtype=np.float64)
    return t < (watermark_s - policy.allowed_lateness_s)
