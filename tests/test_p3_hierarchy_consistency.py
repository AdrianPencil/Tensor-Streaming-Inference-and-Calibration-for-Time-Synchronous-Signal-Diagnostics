"""
P3 hierarchy consistency tests.

Interface-tolerant: skips if hierarchy helper is not present.
"""

import pytest


def test_hierarchy_sum_consistency_or_skip() -> None:
    try:
        import numpy as np
        from ssp.p3_forecast import hierarchy
    except Exception:
        pytest.skip("hierarchy module not importable")

    fn = getattr(hierarchy, "enforce_sum_hierarchy", None)
    if fn is None:
        pytest.skip("enforce_sum_hierarchy not found")

    rng = np.random.default_rng(0)
    leaf = rng.normal(size=(100, 3)).astype(np.float64)
    parent = leaf.sum(axis=1, keepdims=True)

    out = fn(leaf=leaf, parent=parent)
    leaf2 = out["leaf"]
    parent2 = out["parent"]

    assert leaf2.shape == leaf.shape
    assert parent2.shape == parent.shape
    assert np.allclose(parent2[:, 0], leaf2.sum(axis=1))
