"""
ssp.viz.plots_calibration

Calibration plots for scores and probabilities.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_reliability_curve"]


def plot_reliability_curve(prob: NDArray[np.float64], y: NDArray[np.bool_], n_bins: int = 10, title: str = "") -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires optional dependency: matplotlib") from exc

    p = np.asarray(prob, dtype=np.float64)
    yy = np.asarray(y, dtype=np.bool_).astype(np.float64)
    if p.ndim != 1 or yy.ndim != 1 or p.size != yy.size:
        raise ValueError("prob and y must be 1D arrays of equal length")

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, int(n_bins) - 1)

    p_bin = np.zeros(int(n_bins), dtype=np.float64)
    y_bin = np.zeros(int(n_bins), dtype=np.float64)
    cnt = np.zeros(int(n_bins), dtype=np.float64)

    np.add.at(p_bin, idx, p)
    np.add.at(y_bin, idx, yy)
    np.add.at(cnt, idx, 1.0)

    mask = cnt > 0.0
    p_bin[mask] /= cnt[mask]
    y_bin[mask] /= cnt[mask]

    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    ax.plot(p_bin[mask], y_bin[mask], marker="o")
    if title:
        ax.set_title(title)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    return fig, ax
