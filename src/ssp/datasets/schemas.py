"""
ssp.datasets.schemas

Dataset schema containers for synthetic and optional external datasets.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["SyntheticDataset"]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Vectorized synthetic dataset with ground truth labels."""

    t_event_s: NDArray[np.float64]
    y: NDArray[np.float64]
    sensor_id: NDArray[np.int64]
    is_anomaly: NDArray[np.bool_]
    cause: NDArray[np.int64]
    meta: dict[str, Any]
