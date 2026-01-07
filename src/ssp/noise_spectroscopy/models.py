"""
ssp.noise_spectroscopy.models

Parametric PSD models used for system identification.

We provide a minimal set:
- white noise: S(f) = s0
- 1/f: S(f) = a / f^alpha
- Lorentzian: S(f) = a / (1 + (f/fc)^2)
- RTN-like (same functional shape as Lorentzian with different naming)

All models are vectorized and return float64 arrays.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ModelName",
    "WhiteParams",
    "OneFParams",
    "LorentzianParams",
    "psd_white",
    "psd_one_f",
    "psd_lorentzian",
    "psd_model",
]

ModelName = Literal["white", "one_f", "lorentzian", "rtn"]


@dataclass(frozen=True, slots=True)
class WhiteParams:
    s0: float


@dataclass(frozen=True, slots=True)
class OneFParams:
    a: float
    alpha: float
    f_ref_hz: float = 1.0


@dataclass(frozen=True, slots=True)
class LorentzianParams:
    a: float
    fc_hz: float
    s0: float = 0.0


def psd_white(f_hz: NDArray[np.float64], p: WhiteParams) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=np.float64)
    return np.full_like(f, fill_value=float(p.s0), dtype=np.float64)


def psd_one_f(f_hz: NDArray[np.float64], p: OneFParams, eps: float = 1e-24) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=np.float64)
    fr = max(float(p.f_ref_hz), eps)
    x = np.maximum(f / fr, eps)
    return (float(p.a) / (x ** float(p.alpha))).astype(np.float64, copy=False)


def psd_lorentzian(f_hz: NDArray[np.float64], p: LorentzianParams, eps: float = 1e-24) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=np.float64)
    fc = max(float(p.fc_hz), eps)
    denom = 1.0 + (f / fc) ** 2
    return (float(p.s0) + float(p.a) / denom).astype(np.float64, copy=False)


def psd_model(f_hz: NDArray[np.float64], name: ModelName, params: object) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=np.float64)
    if name == "white":
        if not isinstance(params, WhiteParams):
            raise TypeError("white expects WhiteParams")
        return psd_white(f, params)
    if name == "one_f":
        if not isinstance(params, OneFParams):
            raise TypeError("one_f expects OneFParams")
        return psd_one_f(f, params)
    if name in {"lorentzian", "rtn"}:
        if not isinstance(params, LorentzianParams):
            raise TypeError("lorentzian/rtn expects LorentzianParams")
        return psd_lorentzian(f, params)
    raise ValueError(f"Unknown model name: {name}")
