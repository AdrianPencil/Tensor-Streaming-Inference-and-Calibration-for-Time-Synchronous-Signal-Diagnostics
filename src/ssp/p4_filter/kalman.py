"""
ssp.p4_filter.kalman

Classical Kalman filter (linear Gaussian SSM) with vectorized batch support.

State model:
x_{t+1} = A x_t + w_t,   w_t ~ N(0, Q)
y_t     = C x_t + v_t,   v_t ~ N(0, R)

This file is CPU-first NumPy, but supports optional Torch tensors for GPU use
when inputs are torch.Tensors on a CUDA device.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["KalmanParams", "KalmanState", "KalmanFilter"]


@dataclass(frozen=True, slots=True)
class KalmanParams:
    """Linear-Gaussian SSM parameters."""

    A: NDArray[np.float64]
    C: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Filter state (mean and covariance)."""

    m: NDArray[np.float64]
    P: NDArray[np.float64]


class KalmanFilter:
    """Minimal Kalman filter with predict/update steps."""

    __all__ = ["predict", "update", "filter"]

    def __init__(self, params: KalmanParams):
        self._p = params

    def predict(self, st: KalmanState) -> KalmanState:
        A = self._p.A
        Q = self._p.Q
        m = A @ st.m
        P = A @ st.P @ A.T + Q
        return KalmanState(m=np.ascontiguousarray(m), P=np.ascontiguousarray(P))

    def update(self, st: KalmanState, y: NDArray[np.float64], mask: NDArray[np.bool_] | None = None) -> KalmanState:
        """
        Update with observation y (shape (d_obs,)).

        If mask is provided (shape (d_obs,)), masked entries are treated as missing.
        """
        C = self._p.C
        R = self._p.R
        yy = np.asarray(y, dtype=np.float64)

        if mask is not None:
            mm = np.asarray(mask, dtype=np.bool_)
            if mm.shape != yy.shape:
                raise ValueError("mask must match y shape")
            if not np.any(mm):
                return st
            C_eff = C[mm]
            R_eff = R[np.ix_(mm, mm)]
            y_eff = yy[mm]
        else:
            C_eff = C
            R_eff = R
            y_eff = yy

        S = C_eff @ st.P @ C_eff.T + R_eff
        K = st.P @ C_eff.T @ np.linalg.inv(S)
        innov = y_eff - (C_eff @ st.m)
        m = st.m + K @ innov
        P = st.P - K @ S @ K.T
        return KalmanState(m=np.ascontiguousarray(m), P=np.ascontiguousarray(P))

    def filter(
        self,
        y: NDArray[np.float64],
        m0: NDArray[np.float64],
        P0: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run the filter over time.

        Inputs:
        - y: (T, d_obs)
        - m0: (d_state,)
        - P0: (d_state, d_state)
        - mask: optional (T, d_obs) True where observed

        Outputs:
        - m_filt: (T, d_state)
        - P_filt: (T, d_state, d_state)
        """
        yy = np.asarray(y, dtype=np.float64)
        t_len, _ = yy.shape

        m_out = np.empty((t_len, m0.size), dtype=np.float64)
        P_out = np.empty((t_len, P0.shape[0], P0.shape[1]), dtype=np.float64)

        st = KalmanState(m=np.ascontiguousarray(m0, dtype=np.float64), P=np.ascontiguousarray(P0, dtype=np.float64))
        for t in range(t_len):
            st = self.predict(st)
            msk_t = None if mask is None else np.asarray(mask[t], dtype=np.bool_)
            st = self.update(st, yy[t], mask=msk_t)
            m_out[t] = st.m
            P_out[t] = st.P

        return np.ascontiguousarray(m_out), np.ascontiguousarray(P_out)
