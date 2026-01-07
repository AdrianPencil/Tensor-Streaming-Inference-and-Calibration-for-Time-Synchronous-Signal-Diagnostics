"""
ssp.models.transformer_forecaster

A minimal probabilistic Transformer forecaster.

This is a light encoder-only model that predicts (mean, log_var) for the next step.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["TransformerForecasterSpec", "TransformerForecaster"]


@dataclass(frozen=True, slots=True)
class TransformerForecasterSpec:
    d_in: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class TransformerForecaster(nn.Module):
    """Encoder-only next-step Gaussian forecaster."""

    __all__ = ["forward"]

    def __init__(self, spec: TransformerForecasterSpec):
        super().__init__()
        d_in = int(spec.d_in)
        d_model = int(spec.d_model)

        self._proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(spec.n_heads),
            dim_feedforward=4 * d_model,
            dropout=float(spec.dropout),
            batch_first=True,
        )
        self._enc = nn.TransformerEncoder(enc_layer, num_layers=int(spec.n_layers))
        self._head = nn.Linear(d_model, 2 * d_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, T, d_in)

        returns:
        - mean: (N, d_in)
        - log_var: (N, d_in)
        """
        z = self._proj(x)
        h = self._enc(z)
        last = h[:, -1, :]
        out = self._head(last)
        mean, log_var = torch.chunk(out, chunks=2, dim=-1)
        log_var = torch.clamp(log_var, min=-12.0, max=8.0)
        return mean, log_var
