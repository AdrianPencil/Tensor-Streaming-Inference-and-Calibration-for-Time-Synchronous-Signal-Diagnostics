"""
ssp.streaming.windows

Vectorized window assignment for event-time streams.

We provide:
- tumbling windows: fixed width, non-overlapping
- sliding windows: fixed width with a slide step
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["WindowSpec", "tumbling_window_id", "sliding_window_ids"]


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Window parameters in seconds."""

    width_s: float
    slide_s: float | None = None


def tumbling_window_id(t_event_s: NDArray[np.float64], width_s: float) -> NDArray[np.int64]:
    """Assign each timestamp to a tumbling window index."""
    t = np.asarray(t_event_s, dtype=np.float64)
    return np.floor_divide(t, width_s).astype(np.int64)


def sliding_window_ids(t_event_s: NDArray[np.float64], width_s: float, slide_s: float) -> NDArray[np.int64]:
    """
    Assign each timestamp to its sliding window anchor index.

    Interpretation:
    - window anchors are at k * slide_s
    - a timestamp belongs to anchor k if t in [k*slide_s, k*slide_s + width_s)
    """
    t = np.asarray(t_event_s, dtype=np.float64)
    k = np.floor_divide(t, slide_s).astype(np.int64)
    return k
