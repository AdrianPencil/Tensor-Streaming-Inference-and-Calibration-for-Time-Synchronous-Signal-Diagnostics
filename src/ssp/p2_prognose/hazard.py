"""
ssp.p2_prognose.hazard

A lightweight hazard model built from streaming features.

We keep this intentionally simple and engineering-friendly:
- treat hazard as exponential with rate lambda = exp(b0 + b1 * z_t)
- z_t is a scalar covariate derived from alerts / drift stats / score level
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["HazardSpec", "HazardModel"]


@dataclass(frozen=True, slots=True)
class HazardSpec:
    """Hazard model configuration."""

    l2: float = 1e-3
    max_iter: int = 200


class HazardModel:
    """Exponential hazard with a single scalar covariate."""

    def __init__(self, spec: HazardSpec):
        self._s = spec
        self._b0 = 0.0
        self._b1 = 0.0

    def fit(self, z: NDArray[np.float64], dt_s: NDArray[np.float64], failure: NDArray[np.bool_]) -> None:
        """
        Fit using a simple penalized likelihood.

        Inputs:
        - z: (N,) covariate
        - dt_s: (N,) exposure interval in seconds
        - failure: (N,) True if failure occurred at end of interval
        """
        zz = np.asarray(z, dtype=np.float64)
        dt = np.asarray(dt_s, dtype=np.float64)
        y = np.asarray(failure, dtype=np.bool_).astype(np.float64)

        if zz.ndim != 1 or dt.ndim != 1 or y.ndim != 1 or zz.size != dt.size or zz.size != y.size:
            raise ValueError("z, dt_s, failure must be 1D arrays of equal length")

        b0 = 0.0
        b1 = 0.0
        l2 = float(self._s.l2)

        for _ in range(int(self._s.max_iter)):
            eta = b0 + b1 * zz
            lam = np.exp(eta)

            grad0 = np.sum(y - lam * dt) - l2 * b0
            grad1 = np.sum((y - lam * dt) * zz) - l2 * b1

            h00 = -np.sum(lam * dt) - l2
            h11 = -np.sum(lam * dt * zz * zz) - l2
            h01 = -np.sum(lam * dt * zz)

            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break

            db0 = (grad0 * h11 - grad1 * h01) / det
            db1 = (h00 * grad1 - h01 * grad0) / det

            b0 -= db0
            b1 -= db1

            if max(abs(db0), abs(db1)) < 1e-8:
                break

        self._b0 = float(b0)
        self._b1 = float(b1)

    def hazard_rate(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return hazard rate lambda(z) for each covariate value."""
        zz = np.asarray(z, dtype=np.float64)
        return np.exp(self._b0 + self._b1 * zz).astype(np.float64, copy=False)

    def survival(self, z: NDArray[np.float64], horizon_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """S(t|z) = exp(-lambda(z) * t) for exponential hazard."""
        lam = self.hazard_rate(z)
        tt = np.asarray(horizon_s, dtype=np.float64)
        return np.exp(-lam * tt).astype(np.float64, copy=False)
