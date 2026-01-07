"""
ssp.models.vae_density

A minimal VAE density model for anomaly scoring.

Given a feature vector x, model p(x) via latent z.
Use negative ELBO (or reconstruction NLL) as an anomaly score.
"""

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["VaeSpec", "VaeDensity"]


@dataclass(frozen=True, slots=True)
class VaeSpec:
    d_in: int
    d_latent: int = 16
    d_hidden: int = 64


class VaeDensity(nn.Module):
    """Gaussian VAE with diagonal covariance."""

    __all__ = ["forward"]

    def __init__(self, spec: VaeSpec):
        super().__init__()
        d_in = int(spec.d_in)
        d_h = int(spec.d_hidden)
        d_z = int(spec.d_latent)

        self._enc = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
        )
        self._enc_head = nn.Linear(d_h, 2 * d_z)

        self._dec = nn.Sequential(
            nn.Linear(d_z, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
        )
        self._dec_head = nn.Linear(d_h, 2 * d_in)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (N, d_in)

        returns dict with:
        - recon_mean, recon_log_var
        - z_mean, z_log_var
        - z (reparameterized)
        """
        h = self._enc(x)
        z_params = self._enc_head(h)
        z_mean, z_log_var = torch.chunk(z_params, chunks=2, dim=-1)
        z_log_var = torch.clamp(z_log_var, min=-12.0, max=8.0)

        eps = torch.randn_like(z_mean)
        z = z_mean + eps * torch.exp(0.5 * z_log_var)

        g = self._dec(z)
        x_params = self._dec_head(g)
        recon_mean, recon_log_var = torch.chunk(x_params, chunks=2, dim=-1)
        recon_log_var = torch.clamp(recon_log_var, min=-12.0, max=8.0)

        return {
            "recon_mean": recon_mean,
            "recon_log_var": recon_log_var,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "z": z,
        }
