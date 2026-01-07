"""
ssp.iq_readout.snr

SNR and bandwidth utilities.

This module provides:
- SNR estimation from complex IQ by separating "signal-like" energy from noise-like energy
- ENBW (equivalent noise bandwidth) for common windows
- a lightweight ROC helper from continuous scores
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SnrSpec", "snr_db", "estimate_snr_from_iq", "enbw", "roc_curve_from_scores"]


@dataclass(frozen=True, slots=True)
class SnrSpec:
    """SNR estimation settings."""

    guard_frac: float = 0.1
    eps: float = 1e-12


def snr_db(signal_power: float, noise_power: float, eps: float = 1e-12) -> float:
    """Return SNR in dB from powers."""
    sp = max(float(signal_power), eps)
    npw = max(float(noise_power), eps)
    return 10.0 * float(np.log10(sp / npw))


def estimate_snr_from_iq(x: NDArray[np.complex128], spec: SnrSpec = SnrSpec()) -> dict[str, float]:
    """
    Estimate SNR from IQ samples using a robust percentile-based split.

    Interpretation:
    - high-amplitude samples represent signal+noise
    - low-amplitude samples represent mostly noise

    Output keys:
    - snr_db
    - signal_power
    - noise_power
    """
    xx = np.asarray(x, dtype=np.complex128)
    if xx.ndim != 1 or xx.size < 16:
        raise ValueError("x must be 1D complex array with length >= 16")

    amp2 = (xx.real * xx.real + xx.imag * xx.imag).astype(np.float64, copy=False)

    g = float(np.clip(spec.guard_frac, 0.0, 0.45))
    lo_q = g
    hi_q = 1.0 - g

    lo = float(np.quantile(amp2, lo_q))
    hi = float(np.quantile(amp2, hi_q))

    noise_mask = amp2 <= lo
    sig_mask = amp2 >= hi

    eps = float(spec.eps)
    noise_power = float(np.mean(amp2[noise_mask])) if np.any(noise_mask) else float(np.mean(amp2))
    sigp = float(np.mean(amp2[sig_mask])) if np.any(sig_mask) else float(np.mean(amp2))

    signal_power = max(sigp - noise_power, eps)
    return {
        "snr_db": snr_db(signal_power=signal_power, noise_power=noise_power, eps=eps),
        "signal_power": float(signal_power),
        "noise_power": float(noise_power),
    }


def enbw(w: NDArray[np.float64]) -> float:
    """
    Equivalent noise bandwidth (ENBW) for a real window.

    ENBW = fs * sum(w^2) / (sum(w))^2
    For normalized comparisons, we return the dimensionless factor:
    enbw_factor = sum(w^2) / (sum(w))^2
    """
    ww = np.asarray(w, dtype=np.float64)
    if ww.ndim != 1 or ww.size < 2:
        raise ValueError("w must be a 1D array with length >= 2")
    num = float(np.sum(ww * ww))
    den = float(np.sum(ww)) ** 2
    if den <= 0.0:
        raise ValueError("Window sum must be non-zero")
    return num / den


def roc_curve_from_scores(
    score: NDArray[np.float64],
    is_positive: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute an ROC curve from scores.

    Inputs:
    - score: (N,) higher indicates more positive
    - is_positive: (N,) labels

    Output:
    - fpr: (K,)
    - tpr: (K,)
    - thr: (K,) thresholds used
    """
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(is_positive, dtype=np.bool_)
    if s.ndim != 1 or y.ndim != 1 or s.size != y.size:
        raise ValueError("score and is_positive must be 1D arrays of equal length")

    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]

    tp = np.cumsum(y_sorted.astype(np.int64))
    fp = np.cumsum((~y_sorted).astype(np.int64))

    n_pos = int(tp[-1]) if tp.size > 0 else 0
    n_neg = int(fp[-1]) if fp.size > 0 else 0
    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)

    tpr = tp.astype(np.float64) / float(n_pos)
    fpr = fp.astype(np.float64) / float(n_neg)

    return np.ascontiguousarray(fpr), np.ascontiguousarray(tpr), np.ascontiguousarray(s_sorted)
