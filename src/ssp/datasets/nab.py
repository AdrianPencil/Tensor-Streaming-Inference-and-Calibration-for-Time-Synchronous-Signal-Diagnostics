"""
ssp.datasets.nab

Optional loader wrapper for the NAB dataset.

This module is intentionally thin and optional:
- if the dataset is not present locally, it raises a clear error
- it returns vectorized arrays consistent with replay expectations
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["NabSeries", "load_nab_csv"]


@dataclass(frozen=True, slots=True)
class NabSeries:
    """Minimal NAB series representation."""

    t_event_s: NDArray[np.float64]
    y: NDArray[np.float64]


def load_nab_csv(path: Path) -> NabSeries:
    """
    Load a single NAB CSV file.

    Expected columns:
    - timestamp (ISO8601 or numeric)
    - value (float)

    Timestamps are converted to seconds relative to the first entry if possible.
    """
    import csv
    from datetime import datetime

    ts: list[float] = []
    ys: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            y = float(r["value"])
            t_raw = r["timestamp"]
            try:
                t = float(t_raw)
                ts.append(t)
            except ValueError:
                dt = datetime.fromisoformat(t_raw.replace("Z", "+00:00"))
                ts.append(dt.timestamp())
            ys.append(y)

    t_arr = np.asarray(ts, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)

    if t_arr.size == 0:
        raise ValueError(f"Empty NAB file: {path}")

    t_arr = t_arr - float(t_arr[0])

    return NabSeries(t_event_s=t_arr, y=y_arr)
