"""
ssp.p2_prognose.eval

Coverage-style evaluation for probabilistic RUL summaries.

We implement a minimal calibration check:
- given true failure time T and predicted quantiles q_p
- compute empirical coverage at each p
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["CoverageResult", "coverage_from_quantiles"]


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Empirical coverage vs nominal probability."""

    p: NDArray[np.float64]
    coverage: NDArray[np.float64]


def coverage_from_quantiles(
    t_fail_s: NDArray[np.float64],
    q_s: NDArray[np.float64],
    p: NDArray[np.float64],
) -> CoverageResult:
    """
    Compute empirical P(T <= q_p) for each quantile level.

    Inputs:
    - t_fail_s: (N,)
    - q_s: (N, K)
    - p: (K,)

    Output:
    - coverage: (K,)
    """
    t = np.asarray(t_fail_s, dtype=np.float64)
    q = np.asarray(q_s, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)

    if t.ndim != 1 or q.ndim != 2 or pp.ndim != 1:
        raise ValueError("t_fail_s must be (N,), q_s must be (N,K), p must be (K,)")
    if q.shape[0] != t.size or q.shape[1] != pp.size:
        raise ValueError("Shape mismatch between t_fail_s, q_s, and p")

    cov = np.mean((t[:, None] <= q).astype(np.float64), axis=0)
    return CoverageResult(p=pp, coverage=np.ascontiguousarray(cov, dtype=np.float64))
