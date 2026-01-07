"""
ssp.p1_detect.score_ml

Torch-based scoring path for P1.

This is GPU-accelerated by construction:
- move windows to the requested torch device (cpu/cuda)
- compute a simple, strong baseline likelihood score in torch
- later, this file is the natural home for learned models (LSTM/Transformer/VAE)

Current score:
- per-window AR(1) residual energy computed in torch
"""

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = ["score_ml_windows"]


def score_ml_windows(
    x: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
    device: str = "cpu",
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Compute GPU-capable anomaly scores.

    Inputs:
    - x: (N, T, D) float64
    - mask: optional (N, T, D) bool
    - device: "cpu" or "cuda"
    - eps: numerical stabilizer

    Output:
    - scores: (N,) float64 on CPU
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 3:
        raise ValueError("x must have shape (N, T, D)")
    if xx.shape[1] < 3:
        raise ValueError("T must be >= 3 for scoring")

    dev = torch.device(device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    xt = torch.as_tensor(xx, dtype=torch.float64, device=dev)

    mt = None
    if mask is not None:
        mm = np.asarray(mask, dtype=np.bool_)
        if mm.shape != xx.shape:
            raise ValueError("mask must match x shape")
        mt = torch.as_tensor(mm, dtype=torch.bool, device=dev)

    x0 = xt[:, :-1, :]
    x1 = xt[:, 1:, :]

    if mt is None:
        num = torch.sum(x0 * x1, dim=1)
        den = torch.sum(x0 * x0, dim=1) + eps
        rho = num / den

        pred_last = rho * xt[:, -2, :]
        resid = xt[:, -1, :] - pred_last

        res_all = x1 - rho[:, None, :] * x0
        var = torch.mean(res_all * res_all, dim=1) + eps
        score = torch.sum((resid * resid) / var, dim=1)
        return score.detach().to("cpu").numpy().astype(np.float64, copy=False)

    m0 = mt[:, :-1, :]
    m1 = mt[:, 1:, :]
    m_pair = m0 & m1

    x0m = torch.where(m_pair, x0, torch.zeros_like(x0))
    x1m = torch.where(m_pair, x1, torch.zeros_like(x1))

    num = torch.sum(x0m * x1m, dim=1)
    den = torch.sum(x0m * x0m, dim=1) + eps
    rho = num / den

    last_ok = mt[:, -1, :] & mt[:, -2, :]
    pred_last = rho * xt[:, -2, :]
    resid = torch.where(last_ok, xt[:, -1, :] - pred_last, torch.zeros_like(pred_last))

    res_all = x1 - rho[:, None, :] * x0
    res_all = torch.where(m_pair, res_all, torch.zeros_like(res_all))
    cnt = torch.sum(m_pair.to(torch.float64), dim=1).clamp_min(1.0)
    var = torch.sum(res_all * res_all, dim=1) / cnt + eps

    score = torch.sum((resid * resid) / var, dim=1)
    return score.detach().to("cpu").numpy().astype(np.float64, copy=False)
