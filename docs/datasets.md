# Datasets

SSP supports two dataset families:

## 1) Built-in synthetic generator (recommended)

`ssp.datasets.synthetic` produces event-time streams with:
- latent multivariate dynamics
- multiple anomaly families (mean shifts, variance jumps, correlation breaks, missingness bursts)
- known ground truth labels (for self-validation)

This is the default for tests and the fastest way to validate correctness.

## 2) Optional external datasets

Lightweight loaders are in `ssp.datasets`:
- `nab.py`
- `cmaps.py`
- `m5.py`

External datasets are not committed -place them under `data/external/`.
