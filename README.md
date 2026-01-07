# streaming-signal-platform (SSP)

A streaming + replay platform that emits early-warning anomaly alerts with calibrated false-alarm control and root-cause localization (P1), strengthened by filter/fusion (P4) and extended into failure risk (P2) and workload forecasting (P3) -without duplicating infrastructure.

## What you get

- **Event-time streaming core**: timestamps, watermarks, lateness policy, windowing
- **Deterministic replay**: backfill and identical outputs from the same event log
- **P1 Detect**: scoring (non-ML + ML), sequential triggers, calibration/FA control, RCA, drift
- **P4 Filter**: Kalman/UKF skeleton, missingness/multirate handling, stability diagnostics
- **P2 Prognose**: hazard + RUL distribution summaries from streaming signals
- **P3 Forecast**: lightweight forecasting head + hierarchy consistency tools
- **Contracts/Invariants**: bounds, redundancy, spectral expectations as first-class signals
- **Reproducible runs**: manifests, metadata (hashes, env fingerprint), artifacts layout

## Install

### Developer install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"


Licence:

# Architecture

SSP is an engineering system for streaming signal monitoring with replayable evaluation.

## Data model

A stream is treated as an event log:

- `t_event` (event timestamp)
- `sensor_id` (channel key)
- `value` (float measurement)
- optional labels/metadata for evaluation (synthetic ground truth)

The system processes signals in **event-time windows** and outputs score/alert decisions.

## Layer 1: Platform core

### Event-time and windowing (`ssp.streaming`)
- event time primitives
- watermark tracking
- lateness policy
- tumbling/sliding/session windows

### Deterministic replay (`ssp.streaming.replay`)
- consumes an event log and replays it deterministically
- enables regression tests and auditability

### Datasets (`ssp.datasets`)
- schema utilities
- synthetic generator used for self-validation
- optional wrappers for external datasets

### Contracts (`ssp.contracts`)
- bounds checks
- redundancy checks
- spectral checks
- emits structured failure events

### Workflows (`ssp.workflows`)
- run orchestration
- metadata and hashing
- artifact layout + writers
- minimal report rendering

## Layer 2: Pipelines

### P4 Filter (`ssp.p4_filter`)
Produces a stabilized signal/state representation:
- KF skeleton + missingness handling
- stability/diagnostics hooks

### P1 Detect (`ssp.p1_detect`)
The core early-warning pipeline:
- score stream (non-ML or ML)
- sequential triggers
- calibration / false-alarm control
- RCA
- drift detection and retraining hooks

### P2 Prognose (`ssp.p2_prognose`)
Failure risk layer:
- hazard modeling from stream covariates and alert history
- RUL quantile summaries

### P3 Forecast (`ssp.p3_forecast`)
Forecasting layer:
- baseline forecast head
- hierarchy consistency utilities

## Execution model

- config-driven runs via YAML under `experiments/`
- outputs written to an artifact directory
- metadata captures config hash, environment, and code fingerprint

## Acceleration

- numpy/scipy vectorization is the default
- PyTorch provides GPU acceleration for ML scoring when enabled
- NetworkX is used lightly for RCA grouping
