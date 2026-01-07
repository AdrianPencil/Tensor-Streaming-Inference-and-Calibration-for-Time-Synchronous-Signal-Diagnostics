"""
ssp.contracts.spectral

Spectral-band contracts, built on a PSD estimate.

This module does not mandate a PSD method; it consumes (f, psd) arrays and
checks power constraints in defined bands.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["BandSpec", "SpectralResult", "check_band_power"]


@dataclass(frozen=True, slots=True)
class BandSpec:
    """Define a frequency band and expected power range in that band."""

    name: str
    f_lo_hz: float
    f_hi_hz: float
    p_min: float | None = None
    p_max: float | None = None


@dataclass(frozen=True, slots=True)
class SpectralResult:
    """Band-integrated power and violations."""

    band_power: float
    violation: bool


def check_band_power(
    f_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    band: BandSpec,
) -> SpectralResult:
    """
    Integrate PSD over [f_lo, f_hi] and compare to (p_min, p_max).

    Uses trapezoidal integration in frequency.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if f.ndim != 1 or p.ndim != 1 or f.size != p.size:
        raise ValueError("f_hz and psd must be 1D arrays of same length")

    lo = float(band.f_lo_hz)
    hi = float(band.f_hi_hz)
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return SpectralResult(band_power=0.0, violation=False)

    bp = float(np.trapz(p[m], f[m]))
    v = False
    if band.p_min is not None:
        v |= bp < float(band.p_min)
    if band.p_max is not None:
        v |= bp > float(band.p_max)

    return SpectralResult(band_power=bp, violation=bool(v))
