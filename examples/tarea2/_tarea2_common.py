"""
Common utilities for Tarea 2 / ADA 2: relativistic dynamics.

These scripts are intentionally lightweight and can be placed inside your
relativity project under examples/tarea2/. They use only Python, NumPy,
Matplotlib and SymPy for the symbolic derivation exercise.

Main units:
- Energies in MeV unless otherwise stated.
- c = 1 in particle-physics formulas, so mass means rest energy m c^2.
- SI units are used explicitly for Hawking evaporation energy.
"""

from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

import numpy as np

C_SI = 299_792_458.0             # m/s exact
H_SI = 6.626_070_15e-34          # J s exact
EV_J = 1.602_176_634e-19         # J exact
MEV_J = 1.0e6 * EV_J
PROTON_REST_ENERGY_MEV = 938.272_088_16
ELECTRON_REST_ENERGY_MEV = 0.510_998_950_00

OUTPUT_DIR = Path(__file__).resolve().parent / 'output'


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def savefig(fig, filename: str) -> Path:
    out = ensure_output_dir() / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    return out


def print_header(title: str) -> None:
    bar = '=' * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def gamma_beta(beta: float | np.ndarray) -> float | np.ndarray:
    beta = np.asarray(beta, dtype=float)
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError('beta must satisfy |beta| < 1')
    result = 1.0 / np.sqrt(1.0 - beta**2)
    if result.ndim == 0:
        return float(result)
    return result


def beta_from_kinetic_energy(K_MeV: float | np.ndarray, rest_energy_MeV: float) -> float | np.ndarray:
    """Return beta = v/c from kinetic energy K and rest energy E0."""
    K = np.asarray(K_MeV, dtype=float)
    if np.any(K < 0):
        raise ValueError('Kinetic energy must be non-negative.')
    beta = np.sqrt(1.0 - (rest_energy_MeV / (rest_energy_MeV + K))**2)
    if beta.ndim == 0:
        return float(beta)
    return beta


def kinetic_energy_from_beta(beta: float | np.ndarray, rest_energy_MeV: float) -> float | np.ndarray:
    return (gamma_beta(beta) - 1.0) * rest_energy_MeV


def momentum_pc_from_energy_mass(total_energy_MeV: float, rest_energy_MeV: float) -> float:
    """Return p c in MeV from E^2 = (pc)^2 + (m c^2)^2."""
    value = total_energy_MeV**2 - rest_energy_MeV**2
    if value < -1e-12:
        raise ValueError('Total energy is smaller than rest energy.')
    return math.sqrt(max(value, 0.0))


def mev_to_joule(E_MeV: float) -> float:
    return E_MeV * MEV_J


def joule_to_mev(E_J: float) -> float:
    return E_J / MEV_J


def photon_frequency_from_energy_MeV(E_MeV: float) -> float:
    return mev_to_joule(E_MeV) / H_SI


def velocity_addition_beta(beta_object: float, beta_frame: float) -> float:
    """
    Transform 1D object velocity from the primed/rest frame to lab frame:
        beta_lab = (beta_object + beta_frame)/(1 + beta_object beta_frame)
    """
    return (beta_object + beta_frame) / (1.0 + beta_object * beta_frame)


def fmt_sci(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}e}"


def fmt_beta(beta: float, digits: int = 8) -> str:
    return f"{beta:.{digits}f} c"
