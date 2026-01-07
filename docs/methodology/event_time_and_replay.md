# Event time and replay

SSP is designed for streams where data may arrive:
- late
- out of order
- with gaps or bursts

## Event time vs processing time

- **Event time** is the timestamp carried by the record.
- **Processing time** is when the platform observes it.

SSP uses event time semantics for windowing and evaluation.

## Watermarks and lateness

A watermark is the platform’s best current statement of how far event time has progressed.

- Watermark is monotone non-decreasing.
- Allowed lateness controls how far behind the watermark an event can be and still be accepted.

## Replay determinism

`ssp.streaming.replay` provides a deterministic replay runner:
- the same input log yields the same ordered events
- downstream windows and pipelines produce identical outputs

This supports:
- backfills
- auditability
- reproducible evaluations
