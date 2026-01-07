"""
ssp.p3_forecast.interventions

Intervention logs and simple application hooks.

An intervention is a structured event that may shift level or variance.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Intervention", "apply_level_shifts"]


@dataclass(frozen=True, slots=True)
class Intervention:
    """A simple level-shift intervention."""

    t_index: int
    delta: float


def apply_level_shifts(y: NDArray[np.float64], interventions: list[Intervention]) -> NDArray[np.float64]:
    """Apply step level shifts to a 1D series."""
    yy = np.asarray(y, dtype=np.float64).copy()
    for iv in interventions:
        i = int(iv.t_index)
        if 0 <= i < yy.size:
            yy[i:] += float(iv.delta)
    return np.ascontiguousarray(yy, dtype=np.float64)
