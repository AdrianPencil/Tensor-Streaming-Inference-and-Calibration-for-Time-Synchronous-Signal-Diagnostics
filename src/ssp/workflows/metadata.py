"""
ssp.workflows.metadata

Reproducible run metadata:
- config hash
- environment fingerprint
- code fingerprint (best-effort git commit)
"""

from dataclasses import dataclass
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

__all__ = ["RunMetadata", "hash_payload", "collect_env_fingerprint", "collect_git_fingerprint"]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Immutable run metadata bundle."""

    config_hash: str
    env: dict[str, Any]
    code: dict[str, Any]


def hash_payload(payload: dict[str, Any]) -> str:
    """Stable SHA-256 hash of a JSON-serializable payload."""
    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def collect_env_fingerprint() -> dict[str, Any]:
    """Collect environment details relevant for reproducibility."""
    out: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    try:
        import numpy as np

        out["numpy"] = np.__version__
    except Exception:
        out["numpy"] = None

    try:
        import scipy

        out["scipy"] = scipy.__version__
    except Exception:
        out["scipy"] = None

    try:
        import sklearn

        out["scikit_learn"] = sklearn.__version__
    except Exception:
        out["scikit_learn"] = None

    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        out["torch"] = None
        out["cuda_available"] = False

    return out


def collect_git_fingerprint(repo_root: Path) -> dict[str, Any]:
    """Best-effort git fingerprint for the working tree."""
    import subprocess

    out: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            out["git_commit"] = res.stdout.strip()
        res2 = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res2.returncode == 0:
            out["git_dirty"] = bool(res2.stdout.strip())
    except Exception:
        return out
    return out
