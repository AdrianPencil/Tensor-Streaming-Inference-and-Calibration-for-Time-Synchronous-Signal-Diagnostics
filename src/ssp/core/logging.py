"""
ssp.core.logging

Consistent logging configuration for CLI, notebooks, and tests.
"""

import logging

__all__ = ["configure_logging", "get_logger"]


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a stable, compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
