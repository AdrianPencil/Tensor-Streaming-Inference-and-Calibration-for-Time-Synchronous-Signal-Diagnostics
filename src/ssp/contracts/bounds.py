"""
ssp.contracts.bounds

Vectorized range, finite-value, and monotonicity checks.

Contracts are first-class: they emit structured failures that can be treated
as additional signals in pipelines.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["BoundsSpec", "BoundsResult", "check_bounds"]


@dataclass(frozen=True, slots=True)
class BoundsSpec:
    """Bounds and validity rules for a single scalar signal."""

    name: str
    min_value: float | None = None
    max_value: float | None = None
    require_finite: bool = True
    monotone: Literal["none", "nondecreasing", "nonincreasing"] = "none"


@dataclass(frozen=True, slots=True)
class BoundsResult:
    """Vectorized boolean masks for each violation family."""

    out_of_range: NDArray[np.bool_]
    non_finite: NDArray[np.bool_]
    monotone_violation: NDArray[np.bool_]


def check_bounds(y: NDArray[np.float64], spec: BoundsSpec) -> BoundsResult:
    """
    Check an array y against bounds and validity rules.

    Returns boolean masks of shape (N,).
    """
    yy = np.asarray(y, dtype=np.float64)
    non_finite = ~np.isfinite(yy) if spec.require_finite else np.zeros_like(yy, dtype=np.bool_)

    oor = np.zeros_like(yy, dtype=np.bool_)
    if spec.min_value is not None:
        oor |= yy < float(spec.min_value)
    if spec.max_value is not None:
        oor |= yy > float(spec.max_value)

    mono = np.zeros_like(yy, dtype=np.bool_)
    if spec.monotone != "none":
        dy = np.diff(yy)
        if spec.monotone == "nondecreasing":
            bad = dy < 0.0
        else:
            bad = dy > 0.0
        mono[1:] = bad

    return BoundsResult(
        out_of_range=oor,
        non_finite=non_finite,
        monotone_violation=mono,
    )
