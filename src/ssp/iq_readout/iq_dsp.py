"""
ssp.iq_readout.iq_dsp

DSP primitives for complex IQ streams.

We provide:
- DC removal
- IQ imbalance correction (gain/phase)
- frequency shift (complex mixing)
- decimation (SciPy)
- matched filtering (FIR convolution)

All functions are vectorized and use contiguous float64/complex128 arrays.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import decimate

__all__ = ["IqCal", "dc_remove", "mix_down", "apply_iq_cal", "decimate_iq", "matched_filter"]


@dataclass(frozen=True, slots=True)
class IqCal:
    """IQ calibration parameters."""

    i_offset: float = 0.0
    q_offset: float = 0.0
    gain: float = 1.0
    phase_rad: float = 0.0


def dc_remove(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Remove DC by subtracting the complex mean."""
    xx = np.asarray(x, dtype=np.complex128)
    return np.ascontiguousarray(xx - np.mean(xx), dtype=np.complex128)


def mix_down(x: NDArray[np.complex128], f_hz: float, fs_hz: float) -> NDArray[np.complex128]:
    """Complex mix down by frequency f_hz at sample rate fs_hz."""
    xx = np.asarray(x, dtype=np.complex128)
    n = np.arange(xx.size, dtype=np.float64)
    w = -2.0 * np.pi * float(f_hz) / float(fs_hz)
    lo = np.exp(1j * w * n).astype(np.complex128, copy=False)
    return np.ascontiguousarray(xx * lo, dtype=np.complex128)


def apply_iq_cal(x: NDArray[np.complex128], cal: IqCal) -> NDArray[np.complex128]:
    """
    Apply simple IQ calibration:
    - subtract offsets in I and Q
    - correct gain imbalance and quadrature phase error (small-angle model)
    """
    xx = np.asarray(x, dtype=np.complex128)
    i = xx.real - float(cal.i_offset)
    q = xx.imag - float(cal.q_offset)

    q = q / float(cal.gain)

    phi = float(cal.phase_rad)
    q_corr = q * np.cos(phi) - i * np.sin(phi)

    out = i + 1j * q_corr
    return np.ascontiguousarray(out.astype(np.complex128, copy=False))


def decimate_iq(x: NDArray[np.complex128], q: int, ftype: str = "fir") -> NDArray[np.complex128]:
    """Decimate complex IQ by integer factor q using SciPy."""
    xx = np.asarray(x, dtype=np.complex128)
    i = decimate(xx.real, q=int(q), ftype=ftype, zero_phase=True).astype(np.float64, copy=False)
    qv = decimate(xx.imag, q=int(q), ftype=ftype, zero_phase=True).astype(np.float64, copy=False)
    return np.ascontiguousarray((i + 1j * qv).astype(np.complex128, copy=False))


def matched_filter(x: NDArray[np.complex128], h: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """FIR matched filter via convolution with time-reversed conjugate template."""
    xx = np.asarray(x, dtype=np.complex128)
    hh = np.asarray(h, dtype=np.complex128)
    hh_m = np.conjugate(hh[::-1])
    y = np.convolve(xx, hh_m, mode="same")
    return np.ascontiguousarray(y.astype(np.complex128, copy=False))
