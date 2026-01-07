"""
ssp.p4_filter.diagnostics

Observability and stability diagnostics for linear state-space models.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SsDiagnostics", "diagnose_ssm"]


@dataclass(frozen=True, slots=True)
class SsDiagnostics:
    """Diagnostics summary for (A, C)."""

    spectral_radius: float
    stable: bool
    observability_rank: int


def diagnose_ssm(A: NDArray[np.float64], C: NDArray[np.float64]) -> SsDiagnostics:
    """
    Diagnose:
    - stability via spectral radius of A
    - observability rank via observability matrix
    """
    AA = np.asarray(A, dtype=np.float64)
    CC = np.asarray(C, dtype=np.float64)
    if AA.ndim != 2 or CC.ndim != 2:
        raise ValueError("A and C must be 2D")

    eig = np.linalg.eigvals(AA)
    rho = float(np.max(np.abs(eig)))
    stable = rho < 1.0

    d_state = AA.shape[0]
    blocks = []
    Ak = np.eye(d_state, dtype=np.float64)
    for _ in range(d_state):
        blocks.append(CC @ Ak)
        Ak = Ak @ AA
    O = np.vstack(blocks)
    rank = int(np.linalg.matrix_rank(O))

    return SsDiagnostics(spectral_radius=rho, stable=stable, observability_rank=rank)
