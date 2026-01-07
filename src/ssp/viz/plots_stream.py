"""
ssp.viz.plots_stream

Stream visualization helpers (matplotlib).

These functions return (fig, ax) and avoid imposing styles or colors.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_stream"]


def plot_stream(t: NDArray[np.float64], y: NDArray[np.float64], title: str = "") -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires optional dependency: matplotlib") from exc

    tt = np.asarray(t, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    fig, ax = plt.subplots()
    ax.plot(tt, yy)
    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    return fig, ax
