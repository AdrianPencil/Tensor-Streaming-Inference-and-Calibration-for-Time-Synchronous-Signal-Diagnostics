"""
ssp.p1_detect.pipeline

End-to-end P1 pipeline runner for vectorized datasets.

This is the glue:
- create windows
- compute scores (non-ML or ML)
- sequential detection + threshold calibration
- RCA and drift hooks
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ssp.p1_detect.calibrate import CalibrateSpec, ScoreCalibrator
from ssp.p1_detect.drift import DriftSpec, KsDriftDetector
from ssp.p1_detect.rca import RcaSpec, rca_topk
from ssp.p1_detect.score import ScoreDispatcher, ScoreSpec
from ssp.p1_detect.sequential import SequentialDetector, SequentialSpec

__all__ = ["P1Spec", "run_p1"]


@dataclass(frozen=True, slots=True)
class P1Spec:
    """P1 pipeline configuration."""

    score: ScoreSpec = ScoreSpec(path="nonml", window_len=64, device="cpu")
    sequential: SequentialSpec = SequentialSpec()
    calibrate: CalibrateSpec = CalibrateSpec(target_fa_rate=1e-3, online_alpha=0.01, use_isotonic=False)
    rca: RcaSpec = RcaSpec(top_k=3, corr_threshold=0.6)
    drift: DriftSpec = DriftSpec()
    warmup_windows: int = 200


def _make_windows(
    y: NDArray[np.float64],
    t_event_s: NDArray[np.float64],
    sensor_id: NDArray[np.int64],
    win_len: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    yy = np.asarray(y, dtype=np.float64)
    tt = np.asarray(t_event_s, dtype=np.float64)
    ss = np.asarray(sensor_id, dtype=np.int64)
    if yy.ndim != 1 or tt.ndim != 1 or ss.ndim != 1 or yy.size != tt.size or yy.size != ss.size:
        raise ValueError("y, t_event_s, sensor_id must be 1D arrays of equal length")

    d = int(np.max(ss) + 1)
    n = int(yy.size)

    order = np.lexsort((tt, ss))
    yy = yy[order]
    tt = tt[order]
    ss = ss[order]

    counts = np.bincount(ss, minlength=d)
    if np.any(counts < win_len):
        raise ValueError("All sensors must have at least win_len samples to window")

    idx0 = np.zeros(d, dtype=np.int64)
    idx0[1:] = np.cumsum(counts[:-1])

    n_win = int(np.min(counts) - win_len + 1)
    x = np.empty((n_win, win_len, d), dtype=np.float64)
    t_out = np.empty(n_win, dtype=np.float64)

    for j in range(d):
        start = int(idx0[j])
        end = int(start + counts[j])
        s_y = yy[start:end]
        s_t = tt[start:end]
        for k in range(n_win):
            seg = s_y[k : k + win_len]
            x[k, :, j] = seg
        if j == 0:
            t_out[:] = s_t[win_len - 1 : win_len - 1 + n_win]

    return np.ascontiguousarray(x), np.ascontiguousarray(t_out), np.arange(d, dtype=np.int64)


def run_p1(
    y: NDArray[np.float64],
    t_event_s: NDArray[np.float64],
    sensor_id: NDArray[np.int64],
    is_anomaly: NDArray[np.bool_] | None,
    spec: P1Spec,
) -> dict[str, object]:
    """
    Run P1 on a vectorized dataset.

    Output dict fields:
    - t_window: (Nw,)
    - score: (Nw,)
    - alert: (Nw,)
    - threshold: float
    - rca_top: list of arrays for alert times
    - drift: (Nw,) bool
    """
    x, t_w, sensors = _make_windows(y, t_event_s, sensor_id, int(spec.score.window_len))
    scorer = ScoreDispatcher(spec.score)

    scores = scorer.score(x, mask=None)

    calib = ScoreCalibrator(spec.calibrate)
    warm = int(spec.warmup_windows)
    if warm < 10 or warm >= scores.size:
        warm = min(max(10, scores.size // 10), scores.size - 1)

    calib.fit_null(scores[:warm])

    det = SequentialDetector(spec.sequential)
    drift_det = KsDriftDetector(spec.drift)

    alert = np.zeros(scores.size, dtype=np.bool_)
    drift = np.zeros(scores.size, dtype=np.bool_)
    rca_list: list[NDArray[np.int64]] = []
    hist_len = 200

    for i in range(scores.size):
        thr = calib.threshold()
        is_alert_thr = bool(scores[i] >= thr)
        is_alert_seq, _ = det.update(float(scores[i]))
        is_alert = is_alert_thr or is_alert_seq
        alert[i] = is_alert

        if not is_alert:
            calib.update_online(float(scores[i]))

        dflag, _ = drift_det.update(float(scores[i]))
        drift[i] = dflag

        if is_alert:
            i0 = max(0, i - hist_len)
            hist = x[i0:i, -1, :] if i > i0 else None
            score_by_sensor = x[i, -1, :]
            r = rca_topk(score_by_sensor=score_by_sensor, history_scores=hist, spec=spec.rca)
            rca_list.append(r.top_sensors)

    return {
        "t_window": t_w,
        "score": scores,
        "alert": alert,
        "threshold": float(calib.threshold()),
        "rca_top": rca_list,
        "drift": drift,
        "sensors": sensors,
        "is_anomaly": is_anomaly,
    }
