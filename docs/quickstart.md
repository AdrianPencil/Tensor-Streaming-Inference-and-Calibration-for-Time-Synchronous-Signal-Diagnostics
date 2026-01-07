# Quickstart

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

python -m ssp.cli run --config experiments/p1_synth_early_warning.yaml --out runs/p1_synth_early_warning

ml:
  enabled: true
  device: cuda

If CUDA is unavailable, SSP automatically falls back to CPU.

## 137) `docs/architecture.md`

**Folder:** `docs/`  
**File:** `architecture.md`

```markdown
# Architecture

SSP is split into two layers:

## 1) Platform core (shared)

Located in:
- `ssp.streaming`: event time, watermarks, windows, replay, sources/sinks
- `ssp.datasets`: schemas and synthetic generators (self-validation)
- `ssp.contracts`: invariants that emit structured failure events
- `ssp.workflows`: orchestration, metadata, artifacts

These pieces are designed to be reused across all pipelines.

## 2) Signal pipelines

- **P4 Filter (`ssp.p4_filter`)**
  - filtering/fusion and missingness handling
  - produces stable state/feature streams for downstream modules

- **P1 Detect (`ssp.p1_detect`)**
  - scoring (non-ML residuals or ML likelihood/forecast NLL)
  - sequential triggers (CUSUM/SR/EWMA-like)
  - calibration/false-alarm control
  - RCA grouping (NetworkX when present)
  - drift detection and retrain triggers

- **P2 Prognose (`ssp.p2_prognose`)**
  - hazard modeling + RUL quantiles from stream covariates and alert history

- **P3 Forecast (`ssp.p3_forecast`)**
  - lightweight forecasting and hierarchy consistency helpers

## Design constraints

- Deterministic: fixed seeds + identical replay outputs
- Vectorized: numpy/scipy first, contiguous float64 arrays
- Accelerated: torch + CUDA when enabled for ML scoring paths
- Minimal: each module is small, testable, and composable