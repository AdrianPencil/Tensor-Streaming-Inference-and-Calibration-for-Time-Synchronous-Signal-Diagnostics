"""
ssp.workflows.artifacts

Artifact layout and simple writers.

Artifacts are structured:
- raw/ (optional)
- processed/ (arrays, events)
- reports/ (markdown/latex)
- metadata.json (run metadata)
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

__all__ = ["ArtifactPaths", "write_json"]


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    root: Path

    def processed_dir(self) -> Path:
        p = self.root / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def reports_dir(self) -> Path:
        p = self.root / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def result_path(self) -> Path:
        return self.root / "processed" / "result.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
