"""
ssp.models.lstm_forecaster

A minimal probabilistic LSTM forecaster.

Given x_{t-L:t-1} predicts distribution for x_t (mean, log_var).
This can be used to compute per-step NLL as an anomaly score.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["LstmForecasterSpec", "LstmForecaster"]


@dataclass(frozen=True, slots=True)
class LstmForecasterSpec:
    d_in: int
    d_hidden: int = 64
    n_layers: int = 1
    dropout: float = 0.0


class LstmForecaster(nn.Module):
    """Sequence-to-one Gaussian forecaster."""

    __all__ = ["forward"]

    def __init__(self, spec: LstmForecasterSpec):
        super().__init__()
        self._d_in = int(spec.d_in)
        self._lstm = nn.LSTM(
            input_size=self._d_in,
            hidden_size=int(spec.d_hidden),
            num_layers=int(spec.n_layers),
            dropout=float(spec.dropout) if int(spec.n_layers) > 1 else 0.0,
            batch_first=True,
        )
        self._head = nn.Linear(int(spec.d_hidden), 2 * self._d_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, T, d_in)

        returns:
        - mean: (N, d_in)
        - log_var: (N, d_in)
        """
        h, _ = self._lstm(x)
        last = h[:, -1, :]
        out = self._head(last)
        mean, log_var = torch.chunk(out, chunks=2, dim=-1)
        log_var = torch.clamp(log_var, min=-12.0, max=8.0)
        return mean, log_var
