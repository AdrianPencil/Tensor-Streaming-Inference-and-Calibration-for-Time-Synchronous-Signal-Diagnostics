"""
ssp.core.types

Shared data structures and type aliases used across the platform.

Design goals:
- explicit shapes and types for "what flows through streaming"
- minimal, stable primitives for easy import across modules
"""

from dataclasses import dataclass
from typing import Any, Literal, NewType, TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    import torch

    TorchTensor: TypeAlias = torch.Tensor
except Exception:  # pragma: no cover
    torch = None
    TorchTensor = Any  # type: ignore[misc]

__all__ = [
    "SensorId",
    "EventId",
    "Timestamp",
    "FloatArray",
    "BoolArray",
    "ArrayLike",
    "Device",
    "Event",
    "LabeledBatch",
]

SensorId = NewType("SensorId", str)
EventId = NewType("EventId", str)
Timestamp = NewType("Timestamp", float)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

ArrayLike: TypeAlias = FloatArray | TorchTensor

Device: TypeAlias = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class Event:
    """Single streaming record with event-time semantics."""

    event_id: EventId
    sensor_id: SensorId
    t_event: Timestamp
    y: float
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LabeledBatch:
    """Vectorized batch container for model scoring and evaluation."""

    t_event: FloatArray
    y: FloatArray
    sensor_id: NDArray[np.object_]
    is_anomaly: BoolArray | None = None
    cause: NDArray[np.object_] | None = None
