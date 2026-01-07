"""
Spectral contract tests.

Validates that PSD-based band checks can flag an injected narrowband tone.
"""

import pytest


def test_spectral_contract_flags_tone_or_skip() -> None:
    try:
        import numpy as np
        from ssp.contracts import spectral
    except Exception:
        pytest.skip("spectral contract not importable")

    fn = getattr(spectral, "check_psd_band", None)
    if fn is None:
        pytest.skip("check_psd_band not found")

    fs = 2000.0
    n = 65536
    t = np.arange(n, dtype=np.float64) / fs
    x = 0.2 * np.sin(2.0 * np.pi * 200.0 * t)
    x += 0.02 * np.random.default_rng(0).normal(size=n)

    out = fn(x=x, fs_hz=fs, band_hz=(150.0, 250.0), max_frac=0.2)

    assert isinstance(out, dict)
    assert bool(out.get("ok")) is False
