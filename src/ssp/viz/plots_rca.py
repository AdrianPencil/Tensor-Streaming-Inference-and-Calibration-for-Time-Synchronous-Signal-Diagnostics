"""
ssp.viz.plots_rca

RCA visualization helpers.

We keep this minimal:
- bar plot of per-sensor scores at an alert time
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_rca_bars"]


def plot_rca_bars(score_by_sensor: NDArray[np.float64], top_k: int = 5, title: str = "") -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires optional dependency: matplotlib") from exc

    s = np.asarray(score_by_sensor, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("score_by_sensor must be 1D")

    k = int(min(max(1, top_k), s.size))
    idx = np.argsort(-s)[:k].astype(np.int64, copy=False)

    fig, ax = plt.subplots()
    ax.bar(np.arange(k), s[idx])
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(int(i)) for i in idx])
    if title:
        ax.set_title(title)
    ax.set_xlabel("sensor")
    ax.set_ylabel("score")
    return fig, ax
