"""
Replay determinism tests.

Given a fixed event log, replay order and yielded values must be deterministic.
"""

import numpy as np

from ssp.streaming.replay import replay_log


def test_replay_log_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 10.0, size=1000).astype(np.float64)
    y = rng.normal(size=1000).astype(np.float64)
    s = rng.integers(0, 8, size=1000, dtype=np.int64)

    log = list(zip(t.tolist(), y.tolist(), s.tolist()))
    a = list(replay_log(log))
    b = list(replay_log(log))

    assert a == b
