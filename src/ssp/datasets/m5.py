"""
ssp.datasets.m5

Optional loader wrapper for the M5 forecasting dataset.

This stays intentionally thin:
- expects user-provided CSVs
- returns a simple, vectorized view for forecasting heads
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["M5Series", "load_m5_sales"]


@dataclass(frozen=True, slots=True)
class M5Series:
    """A minimal stacked series representation for forecasting."""

    item_id: NDArray[np.int64]
    t: NDArray[np.int64]
    y: NDArray[np.float64]


def load_m5_sales(path: Path, value_prefix: str = "d_") -> M5Series:
    """
    Load the wide-format sales CSV (e.g., sales_train_validation.csv).

    Output:
    - item_id: (N_rows * T,)
    - t: (N_rows * T,)
    - y: (N_rows * T,)
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("M5 loader requires optional dependency: pandas") from exc

    df = pd.read_csv(path)
    time_cols = [c for c in df.columns if c.startswith(value_prefix)]
    if not time_cols:
        raise ValueError(f"No time columns starting with '{value_prefix}' in {path}")

    y_wide = df[time_cols].to_numpy(dtype=np.float64, copy=False)
    n_items, t_len = y_wide.shape

    item_id = np.repeat(np.arange(n_items, dtype=np.int64), repeats=t_len)
    t = np.tile(np.arange(t_len, dtype=np.int64), reps=n_items)
    y = np.ascontiguousarray(y_wide.reshape(-1), dtype=np.float64)

    return M5Series(item_id=item_id, t=t, y=y)
