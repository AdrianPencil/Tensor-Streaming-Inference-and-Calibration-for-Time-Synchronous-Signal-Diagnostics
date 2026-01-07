"""
ssp.core.rng

Deterministic randomness control across Python, NumPy, and PyTorch.

This is the backbone for replay determinism and reproducible synthetic data.
"""

import random
from dataclasses import dataclass

import numpy as np

__all__ = ["RngBundle", "make_rng", "seed_everything"]


@dataclass(frozen=True, slots=True)
class RngBundle:
    """Container for RNG handles used across the project."""

    np_rng: np.random.Generator
    py_seed: int


def make_rng(seed: int) -> RngBundle:
    """Create a reproducible RNG bundle."""
    np_rng = np.random.default_rng(seed)
    return RngBundle(np_rng=np_rng, py_seed=seed)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        return
