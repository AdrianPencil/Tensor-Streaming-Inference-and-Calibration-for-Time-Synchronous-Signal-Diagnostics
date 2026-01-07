"""
ssp.p3_forecast.eval

Forecast evaluation metrics.

We implement:
- MAE
- a simple Gaussian-CRPS approximation when mean+std are provided
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = ["ForecastMetrics", "mae", "crps_gaussian"]


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Forecast metrics container."""

    mae: float
    crps: float | None


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean absolute error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return float(np.mean(np.abs(yt - yp)))


def crps_gaussian(y_true: NDArray[np.float64], mu: NDArray[np.float64], sigma: NDArray[np.float64]) -> float:
    """
    CRPS for Gaussian predictive distribution (vectorized).

    CRPS(N(mu,sigma^2); y) = sigma * [ z*(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ]
    where z=(y-mu)/sigma
    """
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if y.shape != m.shape or y.shape != s.shape:
        raise ValueError("Shapes must match")
    s = np.maximum(s, 1e-12)
    z = (y - m) / s
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    crps = s * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))
