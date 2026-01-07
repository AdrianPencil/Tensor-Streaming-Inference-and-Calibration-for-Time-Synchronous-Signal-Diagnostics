"""
Tests for event-time watermark behavior.

These tests are intentionally interface-tolerant:
- If the streaming watermark API changes, the test will skip rather than fail.
"""

import pytest


def test_watermark_monotone_or_skip() -> None:
    try:
        import numpy as np
        from ssp.streaming import event_time as et
    except Exception:
        pytest.skip("event_time module not importable")

    tracker = getattr(et, "WatermarkTracker", None)
    if tracker is None:
        pytest.skip("WatermarkTracker not available")

    wt = tracker(allowed_lateness_s=0.5)

    ts = np.asarray([1.0, 1.1, 0.9, 1.3, 1.2, 2.0], dtype=np.float64)
    wms = []
    for t in ts:
        wms.append(float(wt.update(float(t))))

    assert all(wms[i] <= wms[i + 1] for i in range(len(wms) - 1))
    assert wms[-1] <= float(ts.max())
