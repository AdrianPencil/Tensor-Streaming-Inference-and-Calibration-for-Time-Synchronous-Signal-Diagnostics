"""
ssp.viz.plots_drift

Drift visualization helpers.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_drift_flags"]


def plot_drift_flags(t: NDArray[np.float64], drift: NDArray[np.bool_], title: str = "") -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires optional dependency: matplotlib") from exc

    tt = np.asarray(t, dtype=np.float64)
    dd = np.asarray(drift, dtype=np.bool_)
    if tt.ndim != 1 or dd.ndim != 1 or tt.size != dd.size:
        raise ValueError("t and drift must be 1D arrays of equal length")

    fig, ax = plt.subplots()
    ax.plot(tt, dd.astype(np.float64))
    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("drift flag")
    ax.set_ylim(-0.05, 1.05)
    return fig, ax
