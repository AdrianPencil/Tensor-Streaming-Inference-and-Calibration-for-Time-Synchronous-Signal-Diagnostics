"""
ssp.p1_detect.score_nonml

Non-ML scoring for P1.

This scorer is a strong, deterministic baseline:
- per-window, per-channel AR(1) fit on the context steps
- predict the final step and compute standardized residual energy
- vectorized across windows and channels
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["score_nonml_windows"]


def score_nonml_windows(
    x: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Compute non-ML anomaly scores for windowed sequences.

    Inputs:
    - x: (N, T, D)
    - mask: optional (N, T, D) True where observed
    - eps: numerical stabilizer

    Output:
    - scores: (N,) higher = more anomalous
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 3:
        raise ValueError("x must have shape (N, T, D)")
    n, t_len, d = xx.shape
    if t_len < 3:
        raise ValueError("T must be >= 3 for AR(1) scoring")

    mm = None
    if mask is not None:
        mm = np.asarray(mask, dtype=np.bool_)
        if mm.shape != xx.shape:
            raise ValueError("mask must match x shape")

    x0 = xx[:, :-1, :]
    x1 = xx[:, 1:, :]

    if mm is None:
        num = np.sum(x0 * x1, axis=1)
        den = np.sum(x0 * x0, axis=1) + eps
        rho = num / den

        pred_last = rho * xx[:, -2, :]
        resid = xx[:, -1, :] - pred_last

        res_all = x1 - rho[:, None, :] * x0
        var = np.mean(res_all * res_all, axis=1) + eps
        score = np.sum((resid * resid) / var, axis=1)
        return np.ascontiguousarray(score, dtype=np.float64)

    m0 = mm[:, :-1, :]
    m1 = mm[:, 1:, :]
    m_pair = m0 & m1

    x0m = np.where(m_pair, x0, 0.0)
    x1m = np.where(m_pair, x1, 0.0)

    num = np.sum(x0m * x1m, axis=1)
    den = np.sum(x0m * x0m, axis=1) + eps
    rho = num / den

    last_ok = mm[:, -1, :] & mm[:, -2, :]
    pred_last = rho * xx[:, -2, :]
    resid = np.where(last_ok, xx[:, -1, :] - pred_last, 0.0)

    res_all = x1 - rho[:, None, :] * x0
    res_all = np.where(m_pair, res_all, 0.0)
    cnt = np.sum(m_pair.astype(np.float64), axis=1)
    cnt = np.maximum(cnt, 1.0)
    var = np.sum(res_all * res_all, axis=1) / cnt + eps

    score = np.sum((resid * resid) / var, axis=1)
    return np.ascontiguousarray(score, dtype=np.float64)
