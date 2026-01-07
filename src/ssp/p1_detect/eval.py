"""
ssp.p1_detect.eval

Evaluation metrics for early-warning detection.

We keep the core metrics simple and replayable:
- false alarm rate (per step)
- detection delay distribution for injected anomalies
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["DetectionMetrics", "evaluate_early_detection"]


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Summary metrics for early warning detection."""

    false_alarm_rate: float
    mean_delay: float
    median_delay: float
    n_events: int
    n_anomaly_events: int
    n_alerts: int


def evaluate_early_detection(
    t_event_s: NDArray[np.float64],
    is_anomaly: NDArray[np.bool_],
    alert: NDArray[np.bool_],
) -> DetectionMetrics:
    """
    Evaluate a single boolean alert stream against ground truth anomaly labels.

    Delay:
    - for each anomaly event time, measure time to first subsequent alert
    - if no alert occurs, that anomaly contributes no delay (conservative)
    """
    t = np.asarray(t_event_s, dtype=np.float64)
    y = np.asarray(is_anomaly, dtype=np.bool_)
    a = np.asarray(alert, dtype=np.bool_)
    if t.ndim != 1 or y.ndim != 1 or a.ndim != 1 or t.size != y.size or t.size != a.size:
        raise ValueError("t_event_s, is_anomaly, alert must be 1D arrays of equal length")

    n = int(t.size)
    n_anom = int(np.sum(y))
    n_alerts = int(np.sum(a))
    n_norm = max(n - n_anom, 1)
    fa_rate = float(np.sum(a & ~y)) / float(n_norm)

    alert_idx = np.nonzero(a)[0]
    delays: list[float] = []
    if alert_idx.size > 0:
        for i in np.nonzero(y)[0]:
            j = alert_idx[np.searchsorted(alert_idx, i, side="left") : ][:1]
            if j.size == 1:
                delays.append(float(t[int(j[0])] - t[int(i)]))

    if len(delays) == 0:
        return DetectionMetrics(
            false_alarm_rate=fa_rate,
            mean_delay=float("inf"),
            median_delay=float("inf"),
            n_events=n,
            n_anomaly_events=n_anom,
            n_alerts=n_alerts,
        )

    dd = np.asarray(delays, dtype=np.float64)
    return DetectionMetrics(
        false_alarm_rate=fa_rate,
        mean_delay=float(np.mean(dd)),
        median_delay=float(np.median(dd)),
        n_events=n,
        n_anomaly_events=n_anom,
        n_alerts=n_alerts,
    )
