"""
ssp.noise_spectroscopy.fit

Robust parameter fits for PSD models.

We fit in log-space to stabilize dynamic range:
- minimize r_i = log(psd_i) - log(model_i)
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ssp.noise_spectroscopy.models import (
    LorentzianParams,
    OneFParams,
    WhiteParams,
    psd_lorentzian,
    psd_one_f,
    psd_white,
)

__all__ = ["FitResult", "fit_white", "fit_one_f", "fit_lorentzian"]


@dataclass(frozen=True, slots=True)
class FitResult:
    params: object
    success: bool
    cost: float
    jac_rank: int


def _fit_log_model(
    f_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    theta0: NDArray[np.float64],
    build_params: Callable[[NDArray[np.float64]], object],
    model: Callable[[NDArray[np.float64], object], NDArray[np.float64]],
) -> FitResult:
    f = np.asarray(f_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if f.ndim != 1 or p.ndim != 1 or f.size != p.size:
        raise ValueError("f_hz and psd must be 1D arrays of equal length")

    eps = 1e-24
    log_p = np.log(np.maximum(p, eps))

    def fun(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        prm = build_params(theta)
        m = model(f, prm)
        log_m = np.log(np.maximum(m, eps))
        return (log_p - log_m).astype(np.float64, copy=False)

    res = least_squares(fun, x0=np.asarray(theta0, dtype=np.float64), method="trf")
    prm = build_params(np.asarray(res.x, dtype=np.float64))
    rank = int(np.linalg.matrix_rank(res.jac)) if res.jac.size > 0 else 0
    return FitResult(params=prm, success=bool(res.success), cost=float(res.cost), jac_rank=rank)


def fit_white(f_hz: NDArray[np.float64], psd: NDArray[np.float64]) -> FitResult:
    p = np.asarray(psd, dtype=np.float64)
    s0 = float(np.median(p))
    theta0 = np.asarray([np.log(max(s0, 1e-24))], dtype=np.float64)

    def build(theta: NDArray[np.float64]) -> WhiteParams:
        return WhiteParams(s0=float(np.exp(theta[0])))

    return _fit_log_model(f_hz, psd, theta0, build, lambda f, prm: psd_white(f, prm))


def fit_one_f(f_hz: NDArray[np.float64], psd: NDArray[np.float64], f_ref_hz: float = 1.0) -> FitResult:
    f = np.asarray(f_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    mask = f > 0.0
    f2 = f[mask]
    p2 = p[mask]
    if f2.size < 10:
        raise ValueError("Need enough positive frequencies to fit 1/f model")

    alpha0 = 1.0
    a0 = float(np.median(p2) * (np.median(f2 / max(f_ref_hz, 1e-24)) ** alpha0))
    theta0 = np.asarray([np.log(max(a0, 1e-24)), np.log(max(alpha0, 1e-6))], dtype=np.float64)

    def build(theta: NDArray[np.float64]) -> OneFParams:
        a = float(np.exp(theta[0]))
        alpha = float(np.exp(theta[1]))
        return OneFParams(a=a, alpha=alpha, f_ref_hz=float(f_ref_hz))

    return _fit_log_model(f_hz, psd, theta0, build, lambda ff, prm: psd_one_f(ff, prm))


def fit_lorentzian(f_hz: NDArray[np.float64], psd: NDArray[np.float64]) -> FitResult:
    f = np.asarray(f_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    s0 = float(np.quantile(p, 0.1))
    a0 = float(max(np.max(p) - s0, 1e-24))
    fc0 = float(np.median(f[f > 0.0])) if np.any(f > 0.0) else 1.0

    theta0 = np.asarray([np.log(max(a0, 1e-24)), np.log(max(fc0, 1e-12)), np.log(max(s0, 1e-24))], dtype=np.float64)

    def build(theta: NDArray[np.float64]) -> LorentzianParams:
        a = float(np.exp(theta[0]))
        fc = float(np.exp(theta[1]))
        s0_ = float(np.exp(theta[2]))
        return LorentzianParams(a=a, fc_hz=fc, s0=s0_)

    return _fit_log_model(f_hz, psd, theta0, build, lambda ff, prm: psd_lorentzian(ff, prm))
