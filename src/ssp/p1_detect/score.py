"""
ssp.p1_detect.score

Anomaly score interface used by P1.

This module defines a stable scoring API that can dispatch to:
- non-ML residual-based scoring (score_nonml)
- ML likelihood/density scoring (score_ml)

Downstream modules (sequential triggers, calibration, RCA) consume scores
without caring which path produced them.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["ScorePath", "ScoreSpec", "ScoreDispatcher"]


ScorePath = Literal["nonml", "ml"]


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """How to compute anomaly scores from an input stream."""

    path: ScorePath = "nonml"
    window_len: int = 64
    device: str = "cpu"


class ScoreDispatcher:
    """Dispatch scoring to the selected implementation path."""

    __all__ = ["score"]

    def __init__(self, spec: ScoreSpec):
        self._spec = spec

    def score(self, x: NDArray[np.float64], mask: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """
        Compute anomaly scores.

        Inputs:
        - x: (N, T, D) windowed sequences
        - mask: optional (N, T, D) observed mask

        Output:
        - s: (N,) anomaly score (higher = more anomalous)
        """
        xx = np.asarray(x, dtype=np.float64)
        if xx.ndim != 3:
            raise ValueError("x must have shape (N, T, D)")

        if self._spec.path == "nonml":
            from ssp.p1_detect.score_nonml import score_nonml_windows

            return score_nonml_windows(xx, mask=mask)

        from ssp.p1_detect.score_ml import score_ml_windows

        return score_ml_windows(xx, mask=mask, device=self._spec.device)
