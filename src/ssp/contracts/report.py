"""
ssp.contracts.report

Structured contract failures emitted as platform events.

Contracts are treated as signals:
- violations can feed RCA and drift logic
- violations can be logged and replayed deterministically
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssp.core.types import Event, EventId, SensorId, Timestamp

__all__ = ["ContractFailure", "failures_to_events"]


@dataclass(frozen=True, slots=True)
class ContractFailure:
    """A contract failure at event-time for a specific sensor or group."""

    name: str
    t_event_s: float
    sensor_id: int
    severity: float
    details: dict[str, Any]


def failures_to_events(
    failures: list[ContractFailure],
    event_id_prefix: str = "contract",
) -> list[Event]:
    """
    Convert contract failures to streaming Events.

    The y field is severity, and payload includes structured details.
    """
    out: list[Event] = []
    for i, f in enumerate(failures):
        out.append(
            Event(
                event_id=EventId(f"{event_id_prefix}:{f.name}:{i}"),
                sensor_id=SensorId(str(f.sensor_id)),
                t_event=Timestamp(float(f.t_event_s)),
                y=float(f.severity),
                payload={"name": f.name, "details": f.details},
            )
        )
    return out
