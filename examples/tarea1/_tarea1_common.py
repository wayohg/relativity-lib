"""
Common utilities for Tarea 1: relativistic kinematics.

These examples are intentionally lightweight and depend only on numpy and
matplotlib. They can live inside your relativity project without modifying the
library itself.

Units used in most exercises:
- c = 1 for natural units unless SI units are explicitly required.
- ly and yr are used together so light travels 1 ly/yr.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import math
import numpy as np

C_SI = 299_792_458.0  # m/s
C_NATURAL = 1.0       # ly/yr or generic natural units

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def savefig(fig, filename: str) -> Path:
    out = ensure_output_dir() / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    return out


def gamma_beta(beta: float | np.ndarray) -> float | np.ndarray:
    beta = np.asarray(beta, dtype=float)
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("beta must satisfy |beta| < 1")
    result = 1.0 / np.sqrt(1.0 - beta**2)
    if result.ndim == 0:
        return float(result)
    return result


def lorentz_x_event(t: float, x: float, beta: float, c: float = C_NATURAL) -> Tuple[float, float]:
    """Lorentz transform for a boost along +x: S -> S'."""
    g = gamma_beta(beta)
    tp = g * (t - beta * x / c)
    xp = g * (x - beta * c * t)
    return float(tp), float(xp)


def lorentz_x_event_ct(ct: float, x: float, beta: float) -> Tuple[float, float]:
    """Lorentz transform using ct and x in the same length units."""
    g = gamma_beta(beta)
    ctp = g * (ct - beta * x)
    xp = g * (x - beta * ct)
    return float(ctp), float(xp)


def velocity_transform_1d(u: float, v: float, c: float = C_NATURAL) -> float:
    """Velocity of an object in S' when object has u in S and S' moves at v wrt S."""
    return (u - v) / (1.0 - u * v / c**2)


def velocity_addition_1d(v: float, w: float, c: float = C_NATURAL) -> float:
    """Einstein addition: (v + w)/(1 + vw/c^2)."""
    return (v + w) / (1.0 + v * w / c**2)


def longitudinal_doppler_factor(beta: float, approaching: bool = False) -> float:
    """
    Frequency factor f_obs/f_emit for longitudinal relativistic Doppler.

    approaching=False: observer/source are separating along light direction.
    approaching=True: observer/source are approaching each other.
    """
    if not (0 <= beta < 1):
        raise ValueError("beta must satisfy 0 <= beta < 1")
    if approaching:
        return math.sqrt((1.0 + beta) / (1.0 - beta))
    return math.sqrt((1.0 - beta) / (1.0 + beta))


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
