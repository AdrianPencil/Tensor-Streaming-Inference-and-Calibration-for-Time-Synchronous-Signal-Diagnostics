"""
ssp.p1_detect.drift

Drift detection utilities for score distributions.

We implement a simple two-sample KS test between a reference window and a recent window.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

__all__ = ["DriftSpec", "KsDriftDetector"]


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """Drift detection settings."""

    ref_size: int = 2000
    recent_size: int = 500
    p_value_threshold: float = 1e-3


class KsDriftDetector:
    """Online drift detector on a univariate score stream."""

    def __init__(self, spec: DriftSpec):
        self._s = spec
        self._ref = deque(maxlen=int(spec.ref_size))
        self._recent = deque(maxlen=int(spec.recent_size))

    def update(self, score: float) -> tuple[bool, dict[str, float]]:
        """
        Update buffers and test drift when ready.

        Returns:
        - drift: bool
        - info: p_value and statistic (if computed)
        """
        x = float(score)
        if len(self._ref) < self._ref.maxlen:
            self._ref.append(x)
            return False, {"p_value": 1.0, "stat": 0.0}

        self._recent.append(x)
        if len(self._recent) < self._recent.maxlen:
            return False, {"p_value": 1.0, "stat": 0.0}

        ref = np.asarray(self._ref, dtype=np.float64)
        rec = np.asarray(self._recent, dtype=np.float64)

        stat, p = ks_2samp(ref, rec, alternative="two-sided", method="auto")
        drift = bool(p < float(self._s.p_value_threshold))
        if drift:
            self._ref.clear()
            self._recent.clear()

        return drift, {"p_value": float(p), "stat": float(stat)}
