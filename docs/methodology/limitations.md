# Limitations

SSP is an engineering platform and is intentionally minimal.

## Known limitations

- Default models are baselines, not domain-optimal.
- RCA grouping depends on available structure and signals.
- ML scoring requires sufficient data and monitoring to avoid silent degradation.
- Distributed execution (MPI) is not implemented by default, but can be layered on later.

## Intended use

Use SSP to build:
- reproducible streaming + replay pipelines
- calibrated early-warning detectors
- validated RCA and drift logic

Then specialize the models as needed for the target domain.
