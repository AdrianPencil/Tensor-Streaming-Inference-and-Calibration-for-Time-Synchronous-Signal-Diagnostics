# Early detection metrics

SSP evaluates detection quality with metrics that reward:
- detecting as early as possible
- controlling false alerts

## Core quantities

Given:
- window timestamps `t_window`
- anomaly labels `is_anomaly` (ground truth)
- alerts `alert` (binary decisions)

We report:
- alert rate
- detection rate
- time-to-first-alert during anomaly intervals
- simple delay summaries

The default metric implementation is in `ssp.p1_detect.eval`.

## Why early detection is different

Standard classification metrics can hide delay cost.

Early detection metrics explicitly track:
- whether a detector fires at all
- how quickly it fires once an anomaly begins
