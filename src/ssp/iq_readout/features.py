"""
ssp.iq_readout.features

Feature extraction for IQ streams.

This module converts complex IQ samples into compact, vectorized feature vectors
for downstream scoring, calibration, and monitoring.

The intended contract is simple:
- input: complex IQ array(s)
- output: float64 feature matrix + names
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

__all__ = ["FeatureSpec", "extract_iq_features"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Feature extraction choices."""

    include_higher_moments: bool = True
    eps: float = 1e-12


def extract_iq_features(x: NDArray[np.complex128], spec: FeatureSpec = FeatureSpec()) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """
    Extract a 1D feature vector from complex IQ samples.

    Input:
    - x: (N,) complex128

    Output:
    - feats: (F,) float64
    - names: (F,) strings
    """
    xx = np.asarray(x, dtype=np.complex128)
    if xx.ndim != 1 or xx.size < 4:
        raise ValueError("x must be a 1D complex array with length >= 4")

    i = xx.real.astype(np.float64, copy=False)
    q = xx.imag.astype(np.float64, copy=False)

    amp = np.abs(xx).astype(np.float64, copy=False)
    power = amp * amp
    phase = np.angle(xx).astype(np.float64, copy=False)

    eps = float(spec.eps)

    i_mean = float(np.mean(i))
    q_mean = float(np.mean(q))
    i_std = float(np.std(i) + eps)
    q_std = float(np.std(q) + eps)

    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp) + eps)
    p_mean = float(np.mean(power))
    p_std = float(np.std(power) + eps)

    ph_mean = float(np.mean(phase))
    ph_std = float(np.std(phase) + eps)

    corr_iq = float(np.mean((i - i_mean) * (q - q_mean)) / (i_std * q_std))

    feats = [
        i_mean,
        q_mean,
        i_std,
        q_std,
        corr_iq,
        amp_mean,
        amp_std,
        p_mean,
        p_std,
        ph_mean,
        ph_std,
    ]
    names = [
        "i_mean",
        "q_mean",
        "i_std",
        "q_std",
        "iq_corr",
        "amp_mean",
        "amp_std",
        "power_mean",
        "power_std",
        "phase_mean",
        "phase_std",
    ]

    if spec.include_higher_moments:
        feats.extend(
            [
                float(skew(i)),
                float(skew(q)),
                float(skew(amp)),
                float(kurtosis(i, fisher=True)),
                float(kurtosis(q, fisher=True)),
                float(kurtosis(amp, fisher=True)),
            ]
        )
        names.extend(
            [
                "i_skew",
                "q_skew",
                "amp_skew",
                "i_kurtosis",
                "q_kurtosis",
                "amp_kurtosis",
            ]
        )

    return np.ascontiguousarray(np.asarray(feats, dtype=np.float64)), tuple(names)
