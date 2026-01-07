# Drift and retraining

Real streams change.

SSP explicitly supports drift detection as part of P1.

## Drift signals

- score distribution shift
- residual variance increase
- contract failure rate changes

## Actions

Depending on configuration:
- re-fit calibration thresholds
- trigger ML retraining hooks
- fall back to non-ML scoring if ML confidence degrades

The reference implementation is in `ssp.p1_detect.drift`.
