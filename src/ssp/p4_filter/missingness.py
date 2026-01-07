"""
ssp.p4_filter.missingness

Utilities for missing observations and multirate streams.

We represent missingness via boolean masks:
- mask[t, j] = True means y[t, j] is observed
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["make_observation_mask", "apply_mask_fill"]


def make_observation_mask(y: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Observed where finite."""
    yy = np.asarray(y, dtype=np.float64)
    return np.isfinite(yy)


def apply_mask_fill(y: NDArray[np.float64], mask: NDArray[np.bool_], fill_value: float = 0.0) -> NDArray[np.float64]:
    """
    Return y_filled where missing entries are replaced by fill_value.

    This is useful when feeding tensors into a model that also consumes mask.
    """
    yy = np.asarray(y, dtype=np.float64)
    mm = np.asarray(mask, dtype=np.bool_)
    if yy.shape != mm.shape:
        raise ValueError("mask must match y shape")
    out = yy.copy()
    out[~mm] = float(fill_value)
    return np.ascontiguousarray(out, dtype=np.float64)
