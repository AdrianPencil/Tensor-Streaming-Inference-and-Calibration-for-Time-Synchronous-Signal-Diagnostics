# Hero pipelines

## Hero 1 - P1 early warning on synthetic ground truth

Input:
- synthetic event-time stream with injected anomaly families and known labels

Output:
- anomaly scores
- calibrated alert decisions
- early detection metrics
- RCA top-k sensors/groups

Why it matters:
- correctness can be validated without a massive real dataset

## Hero 2 - replayable evaluation on an offline event log

Input:
- offline dataset represented as an event log (timestamp, sensor, value)
- replay run with event-time semantics

Output:
- identical results on repeat runs
- operational metrics and early detection metrics

## Hero 3 - P4 filtering improves P1 stability

Input:
- noisy multivariate stream with missingness/multirate behavior

Output:
- filtered state estimates
- reduced false alerts and improved detection delay
- stability/observability diagnostics


# Hero pipelines

## Hero 1: P1 early warning on synthetic ground truth

Input:
- synthetic event-time stream
- injected anomaly families with known labels

Output:
- calibrated alerts with a stable false-alarm target
- early-detection metrics (delay-aware)
- RCA top-k attribution

Validation:
- detection delay distribution
- false alarm metrics
- RCA accuracy vs injected cause groups

## Hero 2: Replayable evaluation on an offline event log

Input:
- dataset represented as an event log
- replayed with event-time semantics

Output:
- identical outputs on repeat runs
- regression testing for algorithm changes

## Hero 3: P1 + P4 filtering reduces false alerts

Input:
- noisy multivariate stream with missingness/multirate behavior

Output:
- filtered state estimate
- improved detection stability and lower false alerts
- diagnostics for observability/stability
