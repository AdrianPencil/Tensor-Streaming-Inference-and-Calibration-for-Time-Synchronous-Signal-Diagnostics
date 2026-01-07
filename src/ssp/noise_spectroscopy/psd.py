"""
ssp.noise_spectroscopy.psd

Power spectral density estimation utilities.

We provide:
- Welch PSD with consistent dtype/contiguity
- small validation helpers for window/segment constraints
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch, get_window

__all__ = ["PsdSpec", "welch_psd"]


@dataclass(frozen=True, slots=True)
class PsdSpec:
    """Welch PSD settings."""

    fs_hz: float
    nperseg: int = 1024
    noverlap: int | None = None
    window: str = "hann"
    detrend: str | None = "constant"
    scaling: str = "density"


def welch_psd(x: NDArray[np.float64], spec: PsdSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Welch PSD.

    Input:
    - x: (N,) real-valued signal

    Output:
    - f_hz: (K,)
    - psd: (K,) in units of x^2/Hz when scaling="density"
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1:
        raise ValueError("x must be 1D")
    if int(spec.nperseg) < 8:
        raise ValueError("nperseg must be >= 8")
    if xx.size < int(spec.nperseg):
        raise ValueError("x must have length >= nperseg")

    win = get_window(spec.window, int(spec.nperseg), fftbins=True)
    f, p = welch(
        xx,
        fs=float(spec.fs_hz),
        window=win,
        nperseg=int(spec.nperseg),
        noverlap=int(spec.noverlap) if spec.noverlap is not None else None,
        detrend=spec.detrend,
        return_onesided=True,
        scaling=spec.scaling,
    )
    return np.ascontiguousarray(f, dtype=np.float64), np.ascontiguousarray(p, dtype=np.float64)
