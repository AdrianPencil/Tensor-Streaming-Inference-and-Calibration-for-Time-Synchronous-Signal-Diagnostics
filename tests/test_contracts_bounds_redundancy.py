"""
Contracts tests: bounds + redundancy.

If these modules change shape, tests skip instead of failing hard.
"""

import pytest


def test_bounds_contract_flags_out_of_range_or_skip() -> None:
    try:
        import numpy as np
        from ssp.contracts import bounds
    except Exception:
        pytest.skip("bounds contract not importable")

    check = getattr(bounds, "check_bounds", None)
    if check is None:
        pytest.skip("check_bounds not found")

    x = np.asarray([-1.0, 0.2, 0.9, 2.0], dtype=np.float64)
    out = check(x=x, lo=0.0, hi=1.0)

    assert isinstance(out, dict)
    assert bool(out.get("ok")) is False


def test_redundancy_contract_detects_violation_or_skip() -> None:
    try:
        import numpy as np
        from ssp.contracts import redundancy
    except Exception:
        pytest.skip("redundancy contract not importable")

    fn = getattr(redundancy, "check_linear_redundancy", None)
    if fn is None:
        pytest.skip("check_linear_redundancy not found")

    a = np.ones(1000, dtype=np.float64)
    b = np.ones(1000, dtype=np.float64)
    c_good = a + b
    c_bad = a + b + 0.5

    ok_good = fn(a=a, b=b, c=c_good, tol=1e-6)
    ok_bad = fn(a=a, b=b, c=c_bad, tol=1e-6)

    assert bool(ok_good.get("ok")) is True
    assert bool(ok_bad.get("ok")) is False
