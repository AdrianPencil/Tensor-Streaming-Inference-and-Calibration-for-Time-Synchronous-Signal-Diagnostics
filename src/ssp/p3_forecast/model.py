"""
ssp.p3_forecast.model

Forecasting head for aggregated metrics.

We provide:
- a strong baseline: seasonal naive / last-value
- optional torch forecaster integration later (kept out of this file for minimalism)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ForecastSpec", "forecast_baseline"]


@dataclass(frozen=True, slots=True)
class ForecastSpec:
    """Baseline forecasting settings."""

    horizon: int = 24
    season: int = 24


def forecast_baseline(y: NDArray[np.float64], spec: ForecastSpec) -> NDArray[np.float64]:
    """
    Seasonal naive baseline.

    If enough history exists: y_hat[t+h] = y[t+h-season]
    Else: last-value carry forward.
    """
    yy = np.asarray(y, dtype=np.float64)
    h = int(spec.horizon)
    s = int(spec.season)

    if yy.ndim != 1:
        raise ValueError("y must be 1D")
    if yy.size == 0:
        raise ValueError("y must be non-empty")

    out = np.empty(h, dtype=np.float64)
    if yy.size >= s:
        ref = yy[-s:]
        reps = int(np.ceil(h / s))
        out[:] = np.tile(ref, reps)[:h]
    else:
        out[:] = yy[-1]

    return np.ascontiguousarray(out, dtype=np.float64)
