# Glossary

- **Event time**: timestamp carried by a record.
- **Processing time**: when the platform processes the record.
- **Watermark**: monotone estimate of event-time progress used to finalize windows.
- **Allowed lateness**: maximum delay tolerated for late events.
- **Window**: aggregation region in event time (tumbling/sliding/session).
- **Replay**: deterministic re-processing of an event log.
- **Score**: continuous anomaly score, higher means more anomalous.
- **Calibration**: mapping scores to thresholds or probabilities with stable meaning.
- **False alarm rate (FA)**: frequency of alerts under normal conditions.
- **RCA**: root-cause localization, attribution of alerts to channels/groups.
- **Drift**: distribution shift or regime change over time.
- **Hazard**: instantaneous failure intensity in survival modeling.
- **RUL**: remaining useful life distribution summaries.
- **Forecasting head**: a module predicting future aggregates/metrics.
- **Contracts**: engineered invariants whose violation is a signal.
