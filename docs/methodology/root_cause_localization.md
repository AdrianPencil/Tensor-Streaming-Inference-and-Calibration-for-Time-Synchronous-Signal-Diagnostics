# Root-cause localization (RCA)

Anomaly detection is incomplete without attribution.

SSP provides a lightweight RCA layer that can work with:
- non-ML scorers (residual-based)
- ML scorers (likelihood/forecast-based)

## RCA objective

At alert times, produce:
- ranked sensors (top-k)
- optionally grouped causes

## How SSP does it

Inputs:
- per-sensor residual/score contributions (when available)
- contract violations (bounds, redundancy, spectral)

Output:
- a ranked list of candidate channels/groups

## Graph grouping (optional)

When NetworkX is installed, SSP can build a simple dependency graph:
- sensors as nodes
- edges encode coupling/expected relationships
- groups are communities / connected components

RCA falls back to non-graph ranking when NetworkX is unavailable.
