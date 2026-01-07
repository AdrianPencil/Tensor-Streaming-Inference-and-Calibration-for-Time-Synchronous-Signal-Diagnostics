"""
P1 early detection metrics tests.
"""

import numpy as np

from ssp.p1_detect.eval import evaluate_early_detection


def test_early_detection_metrics_basic() -> None:
    t = np.arange(100, dtype=np.float64)
    is_anom = np.zeros(100, dtype=np.bool_)
    is_anom[40:60] = True

    alert = np.zeros(100, dtype=np.bool_)
    alert[42] = True
    alert[43] = True

    m = evaluate_early_detection(t_window=t, is_anomaly=is_anom, alert=alert)
    assert isinstance(m, dict)
    assert float(m["alert_rate"]) > 0.0
    assert float(m["detection_rate"]) >= 0.0
