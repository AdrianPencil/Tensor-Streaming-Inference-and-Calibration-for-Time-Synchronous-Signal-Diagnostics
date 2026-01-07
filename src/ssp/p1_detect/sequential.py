"""
ssp.p1_detect.sequential

Sequential detectors consuming anomaly scores.

We implement a compact, deployable subset:
- EWMA z-score trigger
- one-sided CUSUM trigger

All state updates are O(1) per time step, vectorized across sensors if needed.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SequentialSpec", "SequentialState", "SequentialDetector"]


@dataclass(frozen=True, slots=True)
class SequentialSpec:
    """Detector settings for score streams."""

    alpha: float = 0.02
    z_threshold: float = 4.0
    cusum_k: float = 0.5
    cusum_h: float = 8.0


@dataclass(frozen=True, slots=True)
class SequentialState:
    """Running state for sequential detection."""

    mu: float
    var: float
    cusum_pos: float


class SequentialDetector:
    """Online detector: update(score) -> alert flag."""

    def __init__(self, spec: SequentialSpec):
        self._s = spec
        self._st = SequentialState(mu=0.0, var=1.0, cusum_pos=0.0)

    def update(self, score: float) -> tuple[bool, dict[str, float]]:
        """
        Update detector state with a new scalar score.

        Returns:
        - alert: bool
        - info: dict with z and cusum
        """
        a = float(self._s.alpha)
        x = float(score)

        mu = (1.0 - a) * self._st.mu + a * x
        dx = x - mu
        var = (1.0 - a) * self._st.var + a * (dx * dx)
        var = max(var, 1e-12)

        z = (x - mu) / np.sqrt(var)

        k = float(self._s.cusum_k)
        h = float(self._s.cusum_h)
        cpos = max(0.0, self._st.cusum_pos + (x - mu) - k)
        alert = (z >= float(self._s.z_threshold)) or (cpos >= h)

        self._st = SequentialState(mu=mu, var=var, cusum_pos=(0.0 if alert else cpos))

        return alert, {"z": float(z), "cusum": float(cpos), "mu": float(mu), "var": float(var)}
