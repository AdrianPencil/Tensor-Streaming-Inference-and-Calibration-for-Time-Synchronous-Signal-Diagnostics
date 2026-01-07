"""
ssp.core.units

Lightweight unit helpers for time and frequency.

We keep this intentionally minimal: enough for consistent conversions and
readable configs without pulling in a full units library.
"""

from dataclasses import dataclass

__all__ = ["TimeScale", "to_seconds", "hz_to_rad_s"]


@dataclass(frozen=True, slots=True)
class TimeScale:
    """Time unit scale factors to seconds."""

    seconds: float = 1.0
    milliseconds: float = 1e-3
    microseconds: float = 1e-6
    nanoseconds: float = 1e-9


def to_seconds(value: float, unit: str) -> float:
    """Convert a numeric time value in `unit` to seconds."""
    scale = TimeScale()
    if not hasattr(scale, unit):
        raise ValueError(f"Unknown time unit: {unit}")
    return value * float(getattr(scale, unit))


def hz_to_rad_s(f_hz: float) -> float:
    """Convert frequency in Hz to angular frequency in rad/s."""
    return 2.0 * 3.141592653589793 * f_hz
