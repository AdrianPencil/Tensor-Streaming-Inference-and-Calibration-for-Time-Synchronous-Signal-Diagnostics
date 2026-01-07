"""
ssp.p2_prognose.rul

Remaining Useful Life (RUL) summaries from a hazard model.

Given survival S(t), define RUL quantiles q_p via:
S(q_p) = 1 - p
For exponential hazard: q_p = -log(1-p)/lambda
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssp.p2_prognose.hazard import HazardModel

__all__ = ["RulSummary", "rul_quantiles"]


@dataclass(frozen=True, slots=True)
class RulSummary:
    """RUL quantile summary."""

    p: NDArray[np.float64]
    q_s: NDArray[np.float64]


def rul_quantiles(model: HazardModel, z: NDArray[np.float64], p: NDArray[np.float64]) -> RulSummary:
    """
    Compute RUL quantiles under exponential hazard.

    Inputs:
    - z: (N,) covariate values
    - p: (K,) probabilities in (0,1), meaning P(T <= q) = p

    Output:
    - q_s: (N, K)
    """
    zz = np.asarray(z, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)
    if zz.ndim != 1 or pp.ndim != 1:
        raise ValueError("z and p must be 1D arrays")

    lam = model.hazard_rate(zz)[:, None]
    pp = np.clip(pp, 1e-12, 1.0 - 1e-12)[None, :]
    q = (-np.log(1.0 - pp)) / lam
    return RulSummary(p=pp.reshape(-1), q_s=np.ascontiguousarray(q, dtype=np.float64))
