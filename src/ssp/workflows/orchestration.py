"""
ssp.workflows.orchestration

Run orchestration for reproducible executions.

This module executes one run at a time:
- load config
- build pipeline bundle
- run workflow
- write artifacts and metadata

Sweeps/resume can be layered on top without changing the core semantics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ssp.core.config import AppConfig, load_config
from ssp.datasets.synthetic import generate_synthetic
from ssp.workflows.artifacts import ArtifactPaths, write_json
from ssp.workflows.metadata import RunMetadata, collect_env_fingerprint, collect_git_fingerprint, hash_payload
from ssp.workflows.pipelines import WorkflowSpec, run_workflow
from ssp.workflows.registry import build_pipeline_bundle

__all__ = ["RunSpec", "run_one"]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Run spec used by orchestration."""

    config_path: Path
    out_dir: Path
    repo_root: Path


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_one(spec: RunSpec) -> Path:
    """
    Execute a single reproducible run and write artifacts.

    Returns:
    - root artifact directory path
    """
    cfg: AppConfig = load_config(spec.config_path)
    bundle = build_pipeline_bundle(cfg)

    ds = generate_synthetic(cfg.synthetic)

    wf_spec = WorkflowSpec(p1=bundle.p1)
    out = run_workflow(
        t_event_s=ds.t_event_s,
        y=ds.y,
        sensor_id=ds.sensor_id,
        is_anomaly=ds.is_anomaly,
        spec=wf_spec,
    )

    artifacts = ArtifactPaths(root=spec.out_dir)
    env = collect_env_fingerprint()
    code = collect_git_fingerprint(spec.repo_root)

    cfg_payload = {
        "synthetic": cfg.synthetic.__dict__,
        "streaming": cfg.streaming.__dict__,
        "ml": cfg.ml.__dict__,
    }
    md = RunMetadata(config_hash=hash_payload(cfg_payload), env=env, code=code)

    write_json(artifacts.metadata_path(), {"config_hash": md.config_hash, "env": md.env, "code": md.code})
    write_json(artifacts.result_path(), _jsonify(out))

    return artifacts.root
