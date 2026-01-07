"""
ssp.datasets.synthetic

Synthetic stream generator with ground truth anomalies.

Core model (lightweight):
- per-sensor AR(1) latent with shared drift
- noisy observations
- injected anomalies with known cause groups

Outputs are vectorized arrays suitable for replay and evaluation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssp.core.config import SyntheticConfig
from ssp.datasets.schemas import SyntheticDataset

__all__ = ["AnomalyFamily", "generate_synthetic", "synth_preview"]


@dataclass(frozen=True, slots=True)
class AnomalyFamily:
    """Parameterization of basic anomaly types used in self-validation."""

    mean_shift: bool = True
    variance_jump: bool = True


def generate_synthetic(cfg: SyntheticConfig) -> SyntheticDataset:
    """
    Generate a synthetic multivariate stream with ground-truth causes.

    cause codes:
    0 = none
    1..n_sensors = sensor-local anomaly
    """
    rng = np.random.default_rng(cfg.seed)

    n = int(cfg.n_steps)
    d = int(cfg.n_sensors)

    t_event = (np.arange(n, dtype=np.float64) * float(cfg.dt_seconds)).astype(np.float64)

    x = np.zeros((n, d), dtype=np.float64)
    eps = rng.normal(loc=0.0, scale=1.0, size=(n, d)).astype(np.float64)
    for i in range(1, n):
        x[i] = cfg.ar_rho * x[i - 1] + eps[i]

    obs = x + rng.normal(loc=0.0, scale=float(cfg.obs_noise_std), size=(n, d)).astype(np.float64)

    is_anom = rng.random(size=(n, d)) < float(cfg.anomaly_rate)
    cause = np.zeros((n, d), dtype=np.int64)

    shift = float(cfg.anomaly_scale)
    for j in range(d):
        idx = np.nonzero(is_anom[:, j])[0]
        if idx.size == 0:
            continue
        obs[idx, j] += shift * rng.choice([-1.0, 1.0], size=idx.size).astype(np.float64)
        cause[idx, j] = j + 1

    sensor_id = np.repeat(np.arange(d, dtype=np.int64)[None, :], repeats=n, axis=0)
    t_event_2d = np.repeat(t_event[:, None], repeats=d, axis=1)

    out = SyntheticDataset(
        t_event_s=t_event_2d.reshape(-1),
        y=obs.reshape(-1),
        sensor_id=sensor_id.reshape(-1),
        is_anomaly=is_anom.reshape(-1),
        cause=cause.reshape(-1),
        meta={
            "n_steps": n,
            "n_sensors": d,
            "dt_seconds": float(cfg.dt_seconds),
            "ar_rho": float(cfg.ar_rho),
            "obs_noise_std": float(cfg.obs_noise_std),
            "anomaly_rate": float(cfg.anomaly_rate),
            "anomaly_scale": float(cfg.anomaly_scale),
            "seed": int(cfg.seed),
        },
    )
    return out


def synth_preview(cfg: SyntheticConfig) -> dict[str, float]:
    """Small summary used by CLI to sanity-check generator behavior."""
    ds = generate_synthetic(cfg)
    frac_anom = float(np.mean(ds.is_anomaly.astype(np.float64)))
    return {
        "n_rows": float(ds.y.size),
        "n_sensors": float(np.max(ds.sensor_id) + 1),
        "frac_anomaly": frac_anom,
        "y_mean": float(np.mean(ds.y)),
        "y_std": float(np.std(ds.y)),
    }
