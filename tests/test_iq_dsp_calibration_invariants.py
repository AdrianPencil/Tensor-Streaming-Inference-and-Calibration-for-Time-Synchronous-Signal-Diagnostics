"""
IQ DSP + calibration invariants.

Checks that DC removal reduces mean offset and that calibration keeps dtype/shape stable.
"""

import numpy as np

from ssp.iq_readout.calibration import fit_iq_calibration
from ssp.iq_readout.iq_dsp import apply_iq_cal, dc_remove


def test_dc_remove_reduces_offset() -> None:
    rng = np.random.default_rng(0)
    n = 200000
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    x = x + (0.1 - 0.05j)

    m0 = complex(np.mean(x))
    x2 = dc_remove(x)
    m1 = complex(np.mean(x2))

    assert abs(m1) < abs(m0)


def test_fit_and_apply_calibration_shape() -> None:
    rng = np.random.default_rng(1)
    n = 100000
    tone = np.exp(1j * 2.0 * np.pi * 0.01 * np.arange(n)).astype(np.complex128)
    x = 0.2 * tone + (rng.normal(scale=0.05, size=n) + 1j * rng.normal(scale=0.05, size=n))
    x = x.astype(np.complex128)

    fit = fit_iq_calibration(x)
    x3 = apply_iq_cal(dc_remove(x), fit.cal)

    assert x3.shape == x.shape
    assert x3.dtype == np.complex128
