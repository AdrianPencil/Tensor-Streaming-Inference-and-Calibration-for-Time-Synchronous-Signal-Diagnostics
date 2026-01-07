"""
ssp.p3_forecast.hierarchy

Hierarchy consistency helpers.

We represent a hierarchy with a summation matrix S:
- bottom series b (n_bottom,)
- all series y = S b  (n_all,)

Consistency means forecasts respect this structure.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Hierarchy", "project_to_consistent"]


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """Linear summation hierarchy."""

    S: NDArray[np.float64]


def project_to_consistent(y_all: NDArray[np.float64], hierarchy: Hierarchy) -> NDArray[np.float64]:
    """
    Project an inconsistent y_all onto the space {S b} via least squares.

    Inputs:
    - y_all: (n_all,)
    - S: (n_all, n_bottom)

    Output:
    - y_consistent: (n_all,)
    """
    y = np.asarray(y_all, dtype=np.float64)
    S = np.asarray(hierarchy.S, dtype=np.float64)
    if y.ndim != 1 or S.ndim != 2 or y.size != S.shape[0]:
        raise ValueError("y_all must be (n_all,), S must be (n_all, n_bottom)")

    b_hat, *_ = np.linalg.lstsq(S, y, rcond=None)
    y_hat = S @ b_hat
    return np.ascontiguousarray(y_hat, dtype=np.float64)
