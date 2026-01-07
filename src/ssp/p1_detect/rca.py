"""
ssp.p1_detect.rca

Root-cause attribution and grouping.

Design:
- take per-sensor scores or residual energies
- build a lightweight dependency graph (optional NetworkX)
- output top-k likely causes and groups
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["RcaSpec", "RcaResult", "rca_topk"]


@dataclass(frozen=True, slots=True)
class RcaSpec:
    """RCA settings."""

    top_k: int = 3
    corr_threshold: float = 0.6


@dataclass(frozen=True, slots=True)
class RcaResult:
    """RCA output for one alerting time step."""

    top_sensors: NDArray[np.int64]
    groups: list[list[int]]


def rca_topk(
    score_by_sensor: NDArray[np.float64],
    history_scores: NDArray[np.float64] | None,
    spec: RcaSpec,
) -> RcaResult:
    """
    Compute top-k sensor causes and groups.

    Inputs:
    - score_by_sensor: (D,) at the alert time
    - history_scores: optional (T_hist, D) recent history used for correlation grouping

    Output:
    - top sensors (k,)
    - groups: connected components of correlated sensors (if history provided)
    """
    s = np.asarray(score_by_sensor, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("score_by_sensor must be (D,)")

    d = s.size
    k = int(min(max(1, spec.top_k), d))
    top = np.argsort(-s)[:k].astype(np.int64, copy=False)

    groups: list[list[int]] = [[int(i)] for i in top.tolist()]
    if history_scores is None:
        return RcaResult(top_sensors=np.ascontiguousarray(top), groups=groups)

    h = np.asarray(history_scores, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != d or h.shape[0] < 5:
        return RcaResult(top_sensors=np.ascontiguousarray(top), groups=groups)

    corr = np.corrcoef(h.T)
    thr = float(spec.corr_threshold)

    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(d))
        idx = np.argwhere(np.triu(np.abs(corr) >= thr, k=1))
        for i, j in idx:
            g.add_edge(int(i), int(j))
        comps = list(nx.connected_components(g))
        groups = [sorted([int(v) for v in c]) for c in comps if len(c) > 1]
        groups.sort(key=len, reverse=True)
    except Exception:
        idx = np.argwhere(np.triu(np.abs(corr) >= thr, k=1))
        parent = list(range(d))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in idx:
            union(int(i), int(j))

        comp_map: dict[int, list[int]] = {}
        for v in range(d):
            r = find(v)
            comp_map.setdefault(r, []).append(v)

        groups = [sorted(v) for v in comp_map.values() if len(v) > 1]
        groups.sort(key=len, reverse=True)

    return RcaResult(top_sensors=np.ascontiguousarray(top), groups=groups)
