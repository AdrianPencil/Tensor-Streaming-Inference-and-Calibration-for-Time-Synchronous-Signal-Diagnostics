# Calibration and false-alarm control

A raw score stream is not useful unless it can be turned into reliable decisions.

SSP treats calibration as a first-class component.

## Target false alarm rate

Given a target `α`, calibration produces a threshold `τ` such that under a null stream:

`P(score ≥ τ) ≈ α`

The default calibrator:
- fits an empirical quantile on null scores
- supports refits/adaptive updates in the drift module

See `ssp.p1_detect.calibrate`.

## Why calibration matters

Without calibration:
- thresholds drift
- alert meaning changes
- evaluation becomes misleading

With calibration:
- alerts have stable operational meaning
- comparisons across detectors become fair
