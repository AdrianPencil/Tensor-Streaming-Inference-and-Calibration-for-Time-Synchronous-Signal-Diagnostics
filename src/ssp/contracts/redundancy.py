"""
ssp.contracts.redundancy

Cross-sensor consistency checks.

Typical use:
- enforce "sensor_a + sensor_b ≈ sensor_c"
- enforce "two redundant sensors agree within tolerance"
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["RedundancySpec", "RedundancyResult", "check_redundancy"]


@dataclass(frozen=True, slots=True)
class RedundancySpec:
    """Defines a redundancy constraint over channels in a vector y_t."""

    name: str
    kind: Literal["diff", "sum_equals"] = "diff"
    a: int = 0
    b: int = 1
    c: int | None = None
    tol_abs: float = 1.0


@dataclass(frozen=True, slots=True)
class RedundancyResult:
    """Residual and violation mask for the constraint over time."""

    residual: NDArray[np.float64]
    violation: NDArray[np.bool_]


def check_redundancy(y: NDArray[np.float64], spec: RedundancySpec) -> RedundancyResult:
    """
    Check redundancy on a multivariate signal y with shape (T, D).

    Returns:
    - residual: (T,)
    - violation: (T,)
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2:
        raise ValueError("y must have shape (T, D)")

    if spec.kind == "diff":
        r = yy[:, spec.a] - yy[:, spec.b]
    elif spec.kind == "sum_equals":
        if spec.c is None:
            raise ValueError("sum_equals requires c index")
        r = yy[:, spec.a] + yy[:, spec.b] - yy[:, spec.c]
    else:
        raise ValueError(f"Unknown redundancy kind: {spec.kind}")

    r = np.ascontiguousarray(r, dtype=np.float64)
    v = np.abs(r) > float(spec.tol_abs)
    return RedundancyResult(residual=r, violation=v)
