"""
ssp.models.base

Shared ML model base interfaces.

The core contract for P1 is a scorer that maps a batch of sequences/features
to anomaly scores (higher = more anomalous), typically via negative log-likelihood.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["ScoreBatch", "ScoreModel"]


@dataclass(frozen=True, slots=True)
class ScoreBatch:
    """Standard batch container for ML scorers."""

    x: NDArray[np.float64]
    mask: NDArray[np.bool_] | None = None


class ScoreModel(Protocol):
    """Protocol for anything that outputs anomaly scores."""

    def score(self, batch: ScoreBatch) -> NDArray[np.float64]:
        """Return anomaly scores of shape (N,)."""
        ...
