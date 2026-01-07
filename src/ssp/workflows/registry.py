"""
ssp.workflows.registry

Config-to-pipeline factory.

This module provides a single place where the system decides:
- scoring path (non-ML vs ML)
- device selection (cpu vs cuda)
- which workflow components to run in the standard pipeline bundle
"""

from dataclasses import dataclass

from ssp.core.config import AppConfig
from ssp.p1_detect.pipeline import P1Spec
from ssp.p1_detect.score import ScoreSpec

__all__ = ["PipelineBundle", "build_pipeline_bundle"]


@dataclass(frozen=True, slots=True)
class PipelineBundle:
    """Resolved specs for the standard run."""

    p1: P1Spec
    device: str


def build_pipeline_bundle(cfg: AppConfig) -> PipelineBundle:
    """
    Build a resolved pipeline bundle from AppConfig.

    This keeps decisions centralized and reproducible.
    """
    dev = str(cfg.ml.device).lower()
    if dev == "cuda":
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"

    score_path = "ml" if bool(cfg.ml.enabled) else "nonml"
    score_spec = ScoreSpec(path=score_path, window_len=64, device=dev)
    p1 = P1Spec(score=score_spec)
    return PipelineBundle(p1=p1, device=dev)
