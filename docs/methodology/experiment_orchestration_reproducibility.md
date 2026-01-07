# Orchestration and reproducible metadata

Reproducibility is a core requirement.

SSP treats each run as an immutable execution with:
- a config hash
- an environment fingerprint
- a code fingerprint (best-effort git commit + dirty flag)
- structured artifacts

## Artifact layout

- `metadata.json`: reproducibility fingerprint
- `processed/result.json`: machine-readable outputs
- `reports/report.md`: human-readable summary

Implementation lives in `ssp.workflows.metadata`, `ssp.workflows.artifacts`, and `ssp.workflows.orchestration`.
