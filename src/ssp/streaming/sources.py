"""
ssp.streaming.sources

Input sources for event logs.

The platform supports:
- simple file-based sources (CSV/Parquet) as a stand-in for message brokers
- deterministic iteration order for reproducible replay
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ssp.core.types import Event, EventId, SensorId, Timestamp

__all__ = ["FileEventSource"]


@dataclass(frozen=True, slots=True)
class FileEventSource:
    """
    File-based source for an event log.

    Expected columns:
    - event_id (str)
    - sensor_id (str)
    - t_event (float seconds)
    - y (float)
    """

    path: Path

    def iter_events(self) -> Iterator[Event]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            yield from self._iter_csv()
            return
        if suffix in {".parquet"}:
            yield from self._iter_parquet()
            return
        raise ValueError(f"Unsupported source file: {self.path}")

    def _iter_csv(self) -> Iterator[Event]:
        import csv

        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        rows.sort(key=lambda r: (float(r["t_event"]), str(r["event_id"])))

        for r in rows:
            yield Event(
                event_id=EventId(str(r["event_id"])),
                sensor_id=SensorId(str(r["sensor_id"])),
                t_event=Timestamp(float(r["t_event"])),
                y=float(r["y"]),
                payload=None,
            )

    def _iter_parquet(self) -> Iterator[Event]:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Parquet support requires optional dependency: pyarrow") from exc

        table = pq.read_table(self.path)
        data = table.to_pydict()

        event_id = np.asarray(data["event_id"], dtype=object)
        sensor_id = np.asarray(data["sensor_id"], dtype=object)
        t_event = np.asarray(data["t_event"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)

        order = np.lexsort((event_id.astype(str), t_event))
        for i in order:
            yield Event(
                event_id=EventId(str(event_id[i])),
                sensor_id=SensorId(str(sensor_id[i])),
                t_event=Timestamp(float(t_event[i])),
                y=float(y[i]),
                payload=None,
            )
