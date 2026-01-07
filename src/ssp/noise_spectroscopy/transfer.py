"""
ssp.noise_spectroscopy.transfer

Empirical transfer-function / frequency-response estimation.

Given input u(t) and output y(t), estimate:
- H(f) = S_yu(f) / S_uu(f)
- coherence gamma^2(f) = |S_yu|^2 / (S_uu S_yy)

This uses Welch/CSD estimators from SciPy.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import csd, welch, get_window

__all__ = ["TransferSpec", "estimate_transfer"]


@dataclass(frozen=True, slots=True)
class TransferSpec:
    fs_hz: float
    nperseg: int = 1024
    noverlap: int | None = None
    window: str = "hann"
    detrend: str | None = "constant"


def estimate_transfer(
    u: NDArray[np.float64],
    y: NDArray[np.float64],
    spec: TransferSpec,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64]]:
    """
    Estimate transfer function from u to y.

    Inputs:
    - u: (N,)
    - y: (N,)

    Outputs:
    - f_hz: (K,)
    - H: (K,) complex128
    - coh: (K,) float64 coherence in [0,1]
    """
    uu = np.asarray(u, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if uu.ndim != 1 or yy.ndim != 1 or uu.size != yy.size:
        raise ValueError("u and y must be 1D arrays of equal length")
    if uu.size < int(spec.nperseg):
        raise ValueError("signals must have length >= nperseg")

    win = get_window(spec.window, int(spec.nperseg), fftbins=True)

    f, suu = welch(
        uu,
        fs=float(spec.fs_hz),
        window=win,
        nperseg=int(spec.nperseg),
        noverlap=int(spec.noverlap) if spec.noverlap is not None else None,
        detrend=spec.detrend,
        return_onesided=True,
        scaling="density",
    )
    _, syy = welch(
        yy,
        fs=float(spec.fs_hz),
        window=win,
        nperseg=int(spec.nperseg),
        noverlap=int(spec.noverlap) if spec.noverlap is not None else None,
        detrend=spec.detrend,
        return_onesided=True,
        scaling="density",
    )
    _, syu = csd(
        yy,
        uu,
        fs=float(spec.fs_hz),
        window=win,
        nperseg=int(spec.nperseg),
        noverlap=int(spec.noverlap) if spec.noverlap is not None else None,
        detrend=spec.detrend,
        return_onesided=True,
        scaling="density",
    )

    eps = 1e-24
    H = syu / (suu + eps)
    coh = (np.abs(syu) ** 2) / ((suu + eps) * (syy + eps))

    return (
        np.ascontiguousarray(f, dtype=np.float64),
        np.ascontiguousarray(H.astype(np.complex128, copy=False)),
        np.ascontiguousarray(np.clip(coh, 0.0, 1.0).astype(np.float64, copy=False)),
    )
