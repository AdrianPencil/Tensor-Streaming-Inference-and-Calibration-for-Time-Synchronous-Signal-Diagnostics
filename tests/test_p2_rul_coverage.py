"""
P2 RUL tests.

Ensures quantiles are ordered and shapes are consistent.
"""

import numpy as np

from ssp.p2_prognose.hazard import HazardModel, HazardSpec
from ssp.p2_prognose.rul import rul_quantiles


def test_rul_quantiles_monotone() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=5000).astype(np.float64)
    dt = np.full_like(z, 0.01)
    failure_like = z > 2.5

    haz = HazardModel(HazardSpec())
    haz.fit(z=z, dt_s=dt, failure=failure_like)

    p = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    q = rul_quantiles(haz, z=z[:20], p=p)

    assert q.q_s.shape == (20, 3)
    assert np.all(q.q_s[:, 0] <= q.q_s[:, 1])
    assert np.all(q.q_s[:, 1] <= q.q_s[:, 2])
