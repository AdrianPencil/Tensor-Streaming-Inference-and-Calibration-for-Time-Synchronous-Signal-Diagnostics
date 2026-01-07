"""
ssp.streaming.sinks

Output sinks for alerts and metrics.

Sinks are intentionally small:
- JSONL for human/debug use
- Parquet for structured storage (optional dependency)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["JsonlSink"]


@dataclass(frozen=True, slots=True)
class JsonlSink:
    """Append-only JSONL sink."""

    path: Path

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
