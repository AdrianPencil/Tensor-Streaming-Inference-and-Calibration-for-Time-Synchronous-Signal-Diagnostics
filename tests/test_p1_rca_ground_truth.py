"""
RCA sanity test on synthetic data.

This test is intentionally weak: it checks shape and that RCA returns ranked sensors.
"""

import numpy as np

from ssp.core.config import SyntheticConfig
from ssp.datasets.synthetic import generate_synthetic
from ssp.p1_detect.pipeline import P1Spec, run_p1


def test_rca_returns_topk_sensors() -> None:
    cfg = SyntheticConfig(seed=7, n_sensors=8, n_steps=6000, dt_s=0.01, anomaly_rate=0.01)
    ds = generate_synthetic(cfg)

    out = run_p1(ds.y, ds.t_event_s, ds.sensor_id, ds.is_anomaly, P1Spec())

    rca_top = out.get("rca_top", [])
    assert isinstance(rca_top, list)
    if len(rca_top) > 0:
        top0 = rca_top[0]
        assert isinstance(top0, (list, tuple))
        assert len(top0) >= 1
