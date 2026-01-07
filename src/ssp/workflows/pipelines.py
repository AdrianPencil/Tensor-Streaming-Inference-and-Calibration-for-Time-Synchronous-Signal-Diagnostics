"""
ssp.workflows.pipelines

Workflow composition.

This module runs the full platform stack in a single reproducible flow:
- P4 filtering (state estimation / denoising) on multivariate streams
- P1 early warning with calibrated alerting + RCA + drift
- P2 hazard + RUL summaries from P1 outputs
- P3 forecasting on aggregated score/alert metrics

All outputs are plain Python/Numpy objects so they can be persisted easily.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssp.p1_detect.pipeline import P1Spec, run_p1
from ssp.p2_prognose.hazard import HazardModel, HazardSpec
from ssp.p2_prognose.rul import rul_quantiles
from ssp.p3_forecast.model import ForecastSpec, forecast_baseline
from ssp.p4_filter.kalman import KalmanFilter, KalmanParams, KalmanState
from ssp.p4_filter.missingness import make_observation_mask, apply_mask_fill

__all__ = ["WorkflowSpec", "run_workflow"]


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Top-level workflow spec for the standard run."""

    p1: P1Spec
    hazard: HazardSpec = HazardSpec()
    forecast: ForecastSpec = ForecastSpec(horizon=24, season=24)
    rul_p: NDArray[np.float64] = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    use_p4_filter: bool = True


def _p4_simple_identity_filter(y_mv: NDArray[np.float64]) -> NDArray[np.float64]:
    yy = np.asarray(y_mv, dtype=np.float64)
    return np.ascontiguousarray(yy, dtype=np.float64)


def _p4_kalman_filter(y_mv: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    A minimal default linear filter.

    This is not a domain-accurate SSM; it provides a stable denoising/tracking
    baseline until a specific model is configured.
    """
    yy = np.asarray(y_mv, dtype=np.float64)
    if yy.ndim != 2:
        raise ValueError("y_mv must have shape (T, D)")

    t_len, d_obs = yy.shape
    d_state = d_obs

    A = np.eye(d_state, dtype=np.float64)
    C = np.eye(d_obs, dtype=np.float64)

    q = 1e-3
    r = 1e-1
    Q = q * np.eye(d_state, dtype=np.float64)
    R = r * np.eye(d_obs, dtype=np.float64)

    kf = KalmanFilter(KalmanParams(A=A, C=C, Q=Q, R=R))

    mask = make_observation_mask(yy)
    y_fill = apply_mask_fill(yy, mask, fill_value=0.0)

    m0 = np.zeros(d_state, dtype=np.float64)
    P0 = np.eye(d_state, dtype=np.float64)

    m_filt, _ = kf.filter(y=y_fill, m0=m0, P0=P0, mask=mask)
    return np.ascontiguousarray(m_filt, dtype=np.float64)


def run_workflow(
    t_event_s: NDArray[np.float64],
    y: NDArray[np.float64],
    sensor_id: NDArray[np.int64],
    is_anomaly: NDArray[np.bool_] | None,
    spec: WorkflowSpec,
) -> dict[str, object]:
    """
    Run full workflow on vectorized stream data.

    Inputs are the flattened representation:
    - t_event_s: (N,)
    - y: (N,)
    - sensor_id: (N,)

    Output keys include:
    - p1: dict from run_p1
    - p2: hazard/rul summaries
    - p3: forecast on aggregated metrics
    """
    p1_out = run_p1(y=y, t_event_s=t_event_s, sensor_id=sensor_id, is_anomaly=is_anomaly, spec=spec.p1)

    score = np.asarray(p1_out["score"], dtype=np.float64)
    alert = np.asarray(p1_out["alert"], dtype=np.bool_)
    t_w = np.asarray(p1_out["t_window"], dtype=np.float64)

    z = score.astype(np.float64, copy=False)
    dt = np.diff(t_w, prepend=t_w[0]).astype(np.float64, copy=False)
    dt = np.maximum(dt, 1e-6)
    failure_like = alert

    haz = HazardModel(spec.hazard)
    haz.fit(z=z, dt_s=dt, failure=failure_like)

    q = rul_quantiles(model=haz, z=z[:], p=np.asarray(spec.rul_p, dtype=np.float64))

    agg = np.asarray(score, dtype=np.float64)
    y_hat = forecast_baseline(agg, spec.forecast)

    return {
        "p1": p1_out,
        "p2": {
            "hazard_rate": haz.hazard_rate(z),
            "rul_p": q.p,
            "rul_q_s": q.q_s,
        },
        "p3": {
            "forecast": y_hat,
        },
    }
