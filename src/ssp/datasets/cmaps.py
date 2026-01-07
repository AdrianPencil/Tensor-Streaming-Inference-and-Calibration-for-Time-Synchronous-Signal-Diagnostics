"""
ssp.datasets.cmaps

Optional loader wrapper for the NASA C-MAPSS turbofan degradation datasets.

This stays lightweight:
- no downloading
- expects user-provided files in data/external
- returns vectorized arrays suitable for replay / forecasting / RUL heads
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["CmapsDataset", "load_cmaps_txt"]


@dataclass(frozen=True, slots=True)
class CmapsDataset:
    """Minimal C-MAPSS representation (unit, cycle, sensors)."""

    unit_id: NDArray[np.int64]
    cycle: NDArray[np.int64]
    x: NDArray[np.float64]
    feature_names: tuple[str, ...]


def load_cmaps_txt(path: Path) -> CmapsDataset:
    """
    Load a C-MAPSS text file (e.g., train_FD001.txt).

    The canonical format is whitespace-separated with columns:
    unit, cycle, op1, op2, op3, sensor1..sensor21

    We return:
    - unit_id: (N,)
    - cycle: (N,)
    - x: (N, D) where D includes ops + sensors
    """
    raw = np.loadtxt(str(path), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError(f"Unexpected C-MAPSS format: {path}")

    unit_id = raw[:, 0].astype(np.int64, copy=False)
    cycle = raw[:, 1].astype(np.int64, copy=False)
    x = np.ascontiguousarray(raw[:, 2:], dtype=np.float64)

    d = x.shape[1]
    feature_names = tuple([f"f{i+1}" for i in range(d)])

    return CmapsDataset(unit_id=unit_id, cycle=cycle, x=x, feature_names=feature_names)
