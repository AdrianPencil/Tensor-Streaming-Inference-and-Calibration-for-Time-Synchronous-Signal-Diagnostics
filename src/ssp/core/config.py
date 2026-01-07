"""
ssp.core.config

Typed configuration objects with optional YAML/JSON loading.

Guiding principle:
- config objects are plain dataclasses (no heavy frameworks)
- loading is strict enough to catch mistakes, but remains lightweight
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SyntheticConfig",
    "StreamingConfig",
    "MlConfig",
    "AppConfig",
    "load_config",
]


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Synthetic stream generation configuration."""

    n_sensors: int = 8
    n_steps: int = 50_000
    dt_seconds: float = 1.0
    ar_rho: float = 0.98
    obs_noise_std: float = 1.0
    anomaly_rate: float = 5e-4
    anomaly_scale: float = 5.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Event-time streaming semantics configuration."""

    allowed_lateness_seconds: float = 5.0
    watermark_lag_seconds: float = 2.0
    window_seconds: float = 60.0
    slide_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class MlConfig:
    """ML configuration (Torch-based path)."""

    enabled: bool = True
    device: str = "cpu"
    batch_size: int = 256
    lr: float = 1e-3


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration used by the CLI and orchestration layer."""

    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    ml: MlConfig = field(default_factory=MlConfig)

    @staticmethod
    def default() -> "AppConfig":
        return AppConfig()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _asdict_cfg(cfg: AppConfig) -> dict[str, Any]:
    return {
        "synthetic": cfg.synthetic.__dict__,
        "streaming": cfg.streaming.__dict__,
        "ml": cfg.ml.__dict__,
    }


def _from_dict(d: dict[str, Any]) -> AppConfig:
    s = d.get("synthetic", {})
    st = d.get("streaming", {})
    ml = d.get("ml", {})
    return AppConfig(
        synthetic=SyntheticConfig(**s),
        streaming=StreamingConfig(**st),
        ml=MlConfig(**ml),
    )


def load_config(path: Path) -> AppConfig:
    """Load AppConfig from JSON or YAML (if PyYAML is available)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        parsed = json.loads(raw)
    elif path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML config requires optional dependency: PyYAML") from exc
        parsed = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    base = _asdict_cfg(AppConfig.default())
    merged = _deep_update(base, parsed if isinstance(parsed, dict) else {})
    return _from_dict(merged)
