"""
ssp.cli

Command-line entry points.

This file stays light: it wires configs and runs high-level actions.
Actual pipeline logic lives in src/ssp/workflows and src/ssp/p1_detect.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ssp.core.config import AppConfig, load_config
from ssp.core.logging import configure_logging
from ssp.core.rng import seed_everything

__all__ = ["main"]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ssp", description="Streaming Signal Platform")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a config file (YAML or JSON). If omitted, uses defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed for deterministic runs (numpy/random/torch).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-config", help="Load config and print the resolved config JSON.")

    synth = sub.add_parser("synth", help="Generate a synthetic dataset preview.")
    synth.add_argument("--out", type=str, default="", help="Optional output JSON path for preview stats.")

    return parser.parse_args(argv)


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(level=args.log_level)

    seed_everything(seed=args.seed, deterministic=True)

    cfg = load_config(Path(args.config)) if args.config else AppConfig.default()

    if args.command == "validate-config":
        payload = json.dumps(_to_jsonable(cfg), indent=2, sort_keys=True)
        print(payload)
        return 0

    if args.command == "synth":
        from ssp.datasets.synthetic import synth_preview

        preview = synth_preview(cfg.synthetic)
        if args.out:
            Path(args.out).write_text(json.dumps(preview, indent=2, sort_keys=True), encoding="utf-8")
        else:
            print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")
