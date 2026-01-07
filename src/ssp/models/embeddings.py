"""
ssp.models.embeddings

Small embedding blocks (used across forecasters and density models).
"""

import torch
from torch import nn

__all__ = ["SensorEmbedding"]


class SensorEmbedding(nn.Module):
    """Embed discrete sensor/channel IDs into a learned vector space."""

    __all__ = ["forward"]

    def __init__(self, n_sensors: int, d_embed: int):
        super().__init__()
        self._emb = nn.Embedding(int(n_sensors), int(d_embed))

    def forward(self, sensor_id: torch.Tensor) -> torch.Tensor:
        """
        sensor_id:
        - shape (N,) or (N, T) integer tensor

        returns:
        - shape (..., d_embed)
        """
        return self._emb(sensor_id)
