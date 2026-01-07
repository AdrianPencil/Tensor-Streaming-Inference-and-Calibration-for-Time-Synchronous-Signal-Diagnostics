"""
Noise spectroscopy model tests.

Validates positivity and rough monotonicity for 1/f, and that Lorentzian rolls off.
"""

import numpy as np

from ssp.noise_spectroscopy.models import (
    LorentzianParams,
    OneFParams,
    WhiteParams,
    psd_lorentzian,
    psd_one_f,
    psd_white,
)


def test_white_model_constant() -> None:
    f = np.asarray([1.0, 10.0, 100.0], dtype=np.float64)
    p = psd_white(f, WhiteParams(s0=2.0))
    assert np.allclose(p, 2.0)


def test_one_f_decreases_with_f() -> None:
    f = np.asarray([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
    p = psd_one_f(f, OneFParams(a=1.0, alpha=1.0, f_ref_hz=1.0))
    assert np.all(p > 0.0)
    assert p[0] > p[-1]


def test_lorentzian_rolloff() -> None:
    f = np.asarray([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
    p = psd_lorentzian(f, LorentzianParams(a=1.0, fc_hz=10.0, s0=0.1))
    assert np.all(p > 0.0)
    assert p[0] > p[-1]
