"""
P4 filtering tests.

Checks that a simple KF reduces RMSE on a random-walk state model.
"""

import numpy as np

from ssp.p4_filter.kalman import KalmanFilter, KalmanParams


def test_kalman_filter_reduces_rmse() -> None:
    rng = np.random.default_rng(0)
    t = 2000
    d = 4

    A = np.eye(d, dtype=np.float64)
    C = np.eye(d, dtype=np.float64)

    q = 1e-3
    r = 2e-2
    Q = q * np.eye(d, dtype=np.float64)
    R = r * np.eye(d, dtype=np.float64)

    x = np.zeros((t, d), dtype=np.float64)
    for k in range(1, t):
        x[k] = x[k - 1] + rng.normal(scale=np.sqrt(q), size=d)

    y = x + rng.normal(scale=np.sqrt(r), size=(t, d))

    kf = KalmanFilter(KalmanParams(A=A, C=C, Q=Q, R=R))
    m_filt, _ = kf.filter(y=y, m0=np.zeros(d, dtype=np.float64), P0=np.eye(d, dtype=np.float64))

    rmse_raw = float(np.sqrt(np.mean((y - x) ** 2)))
    rmse_kf = float(np.sqrt(np.mean((m_filt - x) ** 2)))

    assert rmse_kf < rmse_raw
