"""
ssp.p4_filter.diff_kalman

Differentiable Kalman parameterization for ML-assisted filtering.

This module learns (A, C, Q, R) via simple constrained parameterizations.
It does not implement training loops; it provides a torch.nn.Module that
returns stable matrices usable by p4_filter.kalman or a torch-based filter.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["DiffKalmanSpec", "DiffKalmanParams"]


@dataclass(frozen=True, slots=True)
class DiffKalmanSpec:
    """Dimensions for the learned linear-Gaussian SSM."""

    d_state: int
    d_obs: int


@dataclass(frozen=True, slots=True)
class DiffKalmanParams:
    """Torch tensors for (A, C, Q, R)."""

    A: torch.Tensor
    C: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor


class DiffKalman(nn.Module):
    """Learn a stable A and positive semidefinite Q,R with simple transforms."""

    __all__ = ["forward"]

    def __init__(self, spec: DiffKalmanSpec):
        super().__init__()
        self._d_state = int(spec.d_state)
        self._d_obs = int(spec.d_obs)

        self._A_raw = nn.Parameter(torch.zeros(self._d_state, self._d_state))
        self._C = nn.Parameter(torch.randn(self._d_obs, self._d_state) * 0.05)

        self._Q_raw = nn.Parameter(torch.eye(self._d_state))
        self._R_raw = nn.Parameter(torch.eye(self._d_obs))

    def forward(self) -> DiffKalmanParams:
        """
        Return constrained parameters.

        Stability: A = tanh(A_raw) / scale to keep spectral radius modest.
        PSD: Q = L L^T + eps I, R = L L^T + eps I
        """
        A = torch.tanh(self._A_raw)
        A = A / max(1.0, float(self._d_state))

        Q = self._Q_raw @ self._Q_raw.T
        R = self._R_raw @ self._R_raw.T

        eps_q = 1e-6 * torch.eye(self._d_state, device=Q.device, dtype=Q.dtype)
        eps_r = 1e-6 * torch.eye(self._d_obs, device=R.device, dtype=R.dtype)

        return DiffKalmanParams(A=A, C=self._C, Q=Q + eps_q, R=R + eps_r)
