# Contracts and invariants

Contracts are engineering rules that should hold in normal operation.

SSP treats contract checks as first-class signals that can:
- trigger alerts
- support RCA
- validate data integrity

## Contract types

- **Bounds**: range checks, basic sanity
- **Redundancy**: cross-sensor consistency constraints
- **Spectral**: PSD band limits or notch expectations

Contracts emit structured events via `ssp.contracts.report`.

## Why contracts matter

Contracts help distinguish:
- data quality issues
- instrumentation problems
- genuine system anomalies

They also create interpretable evidence for RCA.
