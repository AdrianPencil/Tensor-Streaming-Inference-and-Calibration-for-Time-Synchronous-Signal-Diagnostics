"""
Calibration tests: null-fit threshold should achieve approximately target FA rate.
"""

import numpy as np

from ssp.p1_detect.calibrate import CalibrateSpec, ScoreCalibrator


def test_null_calibration_hits_target_rate_roughly() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(size=200000).astype(np.float64)

    target = 1e-3
    cal = ScoreCalibrator(CalibrateSpec(target_fa_rate=target))
    cal.fit_null(s)
    thr = float(cal.threshold())

    fa = float(np.mean(s >= thr))
    assert 0.2 * target <= fa <= 5.0 * target
