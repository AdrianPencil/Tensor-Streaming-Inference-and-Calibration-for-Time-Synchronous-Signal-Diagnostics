"""
ssp.iq_readout.calibration

Calibration primitives for IQ streams.

We estimate a simple calibration:
- I/Q DC offsets
- gain imbalance between I and Q
- quadrature phase error (via correlation)

This keeps calibration explainable, vectorized, and stable.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssp.iq_readout.iq_dsp import IqCal

__all__ = ["IqCalFit", "fit_iq_calibration"]


@dataclass(frozen=True, slots=True)
class IqCalFit:
    """Fit result container."""

    cal: IqCal
    diagnostics: dict[str, float]


def fit_iq_calibration(x: NDArray[np.complex128]) -> IqCalFit:
    """
    Fit simple IQ calibration parameters from samples.

    Assumptions:
- the ideal IQ is centered at 0
- phase error shows as correlation between I and Q
- gain imbalance shows as unequal std
    """
    xx = np.asarray(x, dtype=np.complex128)
    i = xx.real.astype(np.float64, copy=False)
    q = xx.imag.astype(np.float64, copy=False)

    i0 = float(np.mean(i))
    q0 = float(np.mean(q))

    ic = i - i0
    qc = q - q0

    si = float(np.std(ic) + 1e-12)
    sq = float(np.std(qc) + 1e-12)
    gain = sq / si

    corr = float(np.mean((ic / si) * (qc / sq)))
    corr = float(np.clip(corr, -0.999, 0.999))
    phase = float(np.arcsin(corr))

    cal = IqCal(i_offset=i0, q_offset=q0, gain=gain, phase_rad=phase)
    diag = {"i_mean": i0, "q_mean": q0, "i_std": si, "q_std": sq, "corr": corr, "phase_rad": phase}
    return IqCalFit(cal=cal, diagnostics=diag)
