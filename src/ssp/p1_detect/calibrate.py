"""
ssp.p1_detect.calibrate

Calibration and false-alarm control utilities for score streams.

We provide:
- offline threshold by target false-alarm rate (quantile of null scores)
- online adaptive threshold via EWMA quantile tracking
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

__all__ = ["CalibrateSpec", "ScoreCalibrator"]


@dataclass(frozen=True, slots=True)
class CalibrateSpec:
    """Calibration settings."""

    target_fa_rate: float = 1e-3
    online_alpha: float = 0.01
    use_isotonic: bool = False


class ScoreCalibrator:
    """Map raw scores to calibrated p-like values and provide thresholds."""

    def __init__(self, spec: CalibrateSpec):
        self._s = spec
        self._thr = float("inf")
        self._q = float("inf")
        self._iso: IsotonicRegression | None = None

    def fit_null(self, scores_null: NDArray[np.float64]) -> None:
        """
        Fit a null-score distribution and set a fixed threshold.

        scores_null should represent "normal" (non-anomalous) conditions.
        """
        s = np.asarray(scores_null, dtype=np.float64)
        if s.ndim != 1 or s.size < 10:
            raise ValueError("scores_null must be a 1D array with enough samples")
        q = 1.0 - float(self._s.target_fa_rate)
        self._thr = float(np.quantile(s, q))
        self._q = self._thr

    def fit_isotonic(self, scores: NDArray[np.float64], is_anomaly: NDArray[np.bool_]) -> None:
        """
        Optional score-to-prob calibration using isotonic regression.

        This is a convenience layer; alerting still uses thresholds from fit_null or update_online.
        """
        if not self._s.use_isotonic:
            return
        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(is_anomaly, dtype=np.bool_).astype(np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("scores and is_anomaly must be 1D arrays of equal length")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        self._iso = iso

    def score_to_prob(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated anomaly probability if isotonic is fitted, else identity-like mapping."""
        s = np.asarray(scores, dtype=np.float64)
        if self._iso is None:
            return np.clip((s - np.min(s)) / (np.ptp(s) + 1e-12), 0.0, 1.0)
        return np.asarray(self._iso.predict(s), dtype=np.float64)

    def threshold(self) -> float:
        """Current threshold used for alerting."""
        return float(self._thr)

    def update_online(self, score_null_like: float) -> float:
        """
        Online quantile tracker for the null score level.

        Feed it scores believed to be mostly normal (e.g., from non-alert periods).
        Returns the updated threshold.
        """
        a = float(self._s.online_alpha)
        x = float(score_null_like)
        if not np.isfinite(self._q):
            self._q = x
        self._q = (1.0 - a) * self._q + a * x
        self._thr = self._q
        return float(self._thr)
