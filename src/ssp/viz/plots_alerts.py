"""
ssp.viz.plots_alerts

Plot alerts over a score stream.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_scores_with_alerts"]


def plot_scores_with_alerts(
    t: NDArray[np.float64],
    score: NDArray[np.float64],
    alert: NDArray[np.bool_],
    threshold: float | None = None,
    title: str = "",
) -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires optional dependency: matplotlib") from exc

    tt = np.asarray(t, dtype=np.float64)
    ss = np.asarray(score, dtype=np.float64)
    aa = np.asarray(alert, dtype=np.bool_)

    fig, ax = plt.subplots()
    ax.plot(tt, ss, label="score")
    if threshold is not None and np.isfinite(float(threshold)):
        ax.axhline(float(threshold), linestyle="--", label="threshold")
    if np.any(aa):
        ax.scatter(tt[aa], ss[aa], label="alerts")
    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("score")
    ax.legend()
    return fig, ax
