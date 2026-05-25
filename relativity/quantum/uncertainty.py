"""
relativity.quantum.uncertainty
==============================

Utilities for uncertainty-principle calculations in introductory
quantum mechanics.

The functions in this module are designed to be compatible with both
numeric values and simple SymPy symbolic expressions, following the
hybrid style used in the rest of the project.

Main topics
-----------
- Energy-time uncertainty
- Position-momentum uncertainty
- Frequency and wavelength linewidth estimates
- Lifetime broadening
- Natural linewidth
- Momentum/velocity spread estimates

Conventions
-----------
The module uses the common lower-bound form

    ΔA ΔB >= ħ / 2

unless a different numerical factor is explicitly provided.
"""

from __future__ import annotations

from typing import Dict, Optional

import math
import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

try:
    from relativity.constants import C
except Exception:  # pragma: no cover
    C = 299_792_458.0

PLANCK = 6.62607015e-34
HBAR = PLANCK / (2.0 * math.pi)
E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


# ---------------------------------------------------------------------
# Small hybrid helpers
# ---------------------------------------------------------------------

def is_symbolic(x) -> bool:
    """Return True if x contains a SymPy expression."""
    if sp is None:
        return False

    if isinstance(x, sp.Basic):
        return True

    if isinstance(x, (list, tuple, np.ndarray)):
        return any(is_symbolic(item) for item in x)

    return False


def simplify(x):
    """Simplify x when symbolic; otherwise return x unchanged."""
    if sp is not None and is_symbolic(x):
        return sp.simplify(x)
    return x


def smart_abs(x):
    """Absolute value compatible with numeric and symbolic values."""
    if is_symbolic(x):
        return sp.Abs(x)
    return abs(float(x))


def _check_positive(value, name: str, allow_zero: bool = False) -> None:
    """Validate positive numeric values while allowing symbolic inputs."""
    if is_symbolic(value):
        return

    value_float = float(value)

    if allow_zero:
        if value_float < 0:
            raise ValueError(f"{name} must be non-negative.")
    else:
        if value_float <= 0:
            raise ValueError(f"{name} must be positive.")


def _check_mass(mass, name: str = "mass") -> None:
    """Validate a positive mass."""
    _check_positive(mass, name=name, allow_zero=False)


def joule_to_ev(energy_joule):
    """Convert energy from joule to electronvolt."""
    return simplify(energy_joule / E_CHARGE)


def ev_to_joule(energy_ev):
    """Convert energy from electronvolt to joule."""
    return simplify(energy_ev * E_CHARGE)


# ---------------------------------------------------------------------
# General uncertainty relations
# ---------------------------------------------------------------------

def uncertainty_lower_bound(delta_a, constant=HBAR, factor: float = 0.5):
    """
    Generic lower bound for a conjugate uncertainty.

    For a relation ΔA ΔB >= factor * constant, this returns:

        ΔB_min = factor * constant / ΔA
    """
    _check_positive(delta_a, "delta_a")
    return simplify(factor * constant / delta_a)


def product_lower_bound(constant=HBAR, factor: float = 0.5):
    """
    Return the lower bound factor * constant for ΔA ΔB.
    """
    return simplify(factor * constant)


def satisfies_uncertainty(delta_a, delta_b, constant=HBAR, factor: float = 0.5, tol: float = 1e-12):
    """
    Check whether ΔA ΔB satisfies the uncertainty lower bound.

    For symbolic inputs this returns a SymPy relational expression.
    For numeric inputs this returns bool.
    """
    product = simplify(delta_a * delta_b)
    bound = product_lower_bound(constant=constant, factor=factor)

    if is_symbolic(product) or is_symbolic(bound):
        return sp.Ge(sp.simplify(product - bound), 0)

    return bool(product + tol >= bound)


# ---------------------------------------------------------------------
# Energy-time uncertainty
# ---------------------------------------------------------------------

def energy_uncertainty_from_time(delta_t, hbar=HBAR, factor: float = 0.5):
    """
    Minimum energy uncertainty from a time uncertainty:

        ΔE >= factor * ħ / Δt
    """
    _check_positive(delta_t, "delta_t")
    return simplify(factor * hbar / delta_t)


def time_uncertainty_from_energy(delta_E, hbar=HBAR, factor: float = 0.5):
    """
    Minimum time uncertainty from an energy uncertainty:

        Δt >= factor * ħ / ΔE
    """
    _check_positive(delta_E, "delta_E")
    return simplify(factor * hbar / delta_E)


def energy_uncertainty_from_lifetime(lifetime, hbar=HBAR, factor: float = 0.5):
    """
    Estimate natural energy uncertainty from the lifetime of a state.

    Default factor=0.5 corresponds to ΔE Δt >= ħ/2.
    Use factor=1.0 for an order-of-magnitude convention ΔE ≈ ħ/τ.
    """
    return energy_uncertainty_from_time(lifetime, hbar=hbar, factor=factor)


def lifetime_from_energy_uncertainty(delta_E, hbar=HBAR, factor: float = 0.5):
    """
    Estimate lifetime from energy uncertainty:

        τ ≈ factor * ħ / ΔE
    """
    return time_uncertainty_from_energy(delta_E, hbar=hbar, factor=factor)


def angular_frequency_uncertainty_from_lifetime(lifetime, factor: float = 0.5):
    """
    Angular-frequency uncertainty from a lifetime:

        Δω ≈ factor / τ
    """
    _check_positive(lifetime, "lifetime")
    return simplify(factor / lifetime)


def frequency_uncertainty_from_lifetime(lifetime, factor: float = 0.5):
    """
    Frequency uncertainty from a lifetime:

        Δf ≈ factor / (2π τ)
    """
    _check_positive(lifetime, "lifetime")
    return simplify(factor / (2.0 * math.pi * lifetime))


def frequency_uncertainty_from_energy(delta_E, h=PLANCK):
    """
    Convert energy uncertainty to frequency uncertainty:

        Δf = ΔE / h
    """
    _check_positive(delta_E, "delta_E")
    return simplify(delta_E / h)


def energy_uncertainty_from_frequency(delta_f, h=PLANCK):
    """
    Convert frequency uncertainty to energy uncertainty:

        ΔE = h Δf
    """
    _check_positive(delta_f, "delta_f")
    return simplify(h * delta_f)


# ---------------------------------------------------------------------
# Position-momentum uncertainty
# ---------------------------------------------------------------------

def momentum_uncertainty_from_position(delta_x, hbar=HBAR, factor: float = 0.5):
    """
    Minimum momentum uncertainty from position uncertainty:

        Δp >= factor * ħ / Δx
    """
    _check_positive(delta_x, "delta_x")
    return simplify(factor * hbar / delta_x)


def position_uncertainty_from_momentum(delta_p, hbar=HBAR, factor: float = 0.5):
    """
    Minimum position uncertainty from momentum uncertainty:

        Δx >= factor * ħ / Δp
    """
    _check_positive(delta_p, "delta_p")
    return simplify(factor * hbar / delta_p)


def velocity_uncertainty_from_position(delta_x, mass=ELECTRON_MASS, hbar=HBAR, factor: float = 0.5):
    """
    Nonrelativistic velocity uncertainty from position uncertainty:

        Δv ≈ Δp / m
    """
    _check_mass(mass)
    delta_p = momentum_uncertainty_from_position(delta_x, hbar=hbar, factor=factor)
    return simplify(delta_p / mass)


def position_uncertainty_from_velocity(delta_v, mass=ELECTRON_MASS, hbar=HBAR, factor: float = 0.5):
    """
    Position uncertainty from velocity uncertainty using Δp ≈ m Δv.
    """
    _check_positive(delta_v, "delta_v")
    _check_mass(mass)
    delta_p = mass * delta_v
    return position_uncertainty_from_momentum(delta_p, hbar=hbar, factor=factor)


def momentum_spread_from_wavelength_spread(wavelength, delta_wavelength, h=PLANCK):
    """
    Approximate momentum spread from wavelength spread.

    For p = h / λ:

        |Δp| ≈ h |Δλ| / λ²
    """
    _check_positive(wavelength, "wavelength")
    _check_positive(delta_wavelength, "delta_wavelength", allow_zero=True)
    return simplify(h * smart_abs(delta_wavelength) / wavelength**2)


def wavelength_spread_from_momentum_spread(wavelength, delta_p, h=PLANCK):
    """
    Approximate wavelength spread from momentum spread.

        |Δλ| ≈ λ² |Δp| / h
    """
    _check_positive(wavelength, "wavelength")
    _check_positive(delta_p, "delta_p", allow_zero=True)
    return simplify(wavelength**2 * smart_abs(delta_p) / h)


# ---------------------------------------------------------------------
# Spectral linewidth helpers
# ---------------------------------------------------------------------

def wavelength_uncertainty_from_frequency(wavelength, delta_f, c=C):
    """
    Approximate wavelength uncertainty from frequency uncertainty.

    Since f = c/λ:

        |Δλ| ≈ λ² |Δf| / c
    """
    _check_positive(wavelength, "wavelength")
    _check_positive(delta_f, "delta_f", allow_zero=True)
    _check_positive(c, "c")
    return simplify(wavelength**2 * smart_abs(delta_f) / c)


def frequency_uncertainty_from_wavelength(wavelength, delta_wavelength, c=C):
    """
    Approximate frequency uncertainty from wavelength uncertainty:

        |Δf| ≈ c |Δλ| / λ²
    """
    _check_positive(wavelength, "wavelength")
    _check_positive(delta_wavelength, "delta_wavelength", allow_zero=True)
    _check_positive(c, "c")
    return simplify(c * smart_abs(delta_wavelength) / wavelength**2)


def wavelength_uncertainty_from_lifetime(wavelength, lifetime, c=C, factor: float = 0.5):
    """
    Estimate natural wavelength linewidth from lifetime.

    Uses:
        Δf ≈ factor / (2π τ)
        |Δλ| ≈ λ² Δf / c
    """
    delta_f = frequency_uncertainty_from_lifetime(lifetime, factor=factor)
    return wavelength_uncertainty_from_frequency(wavelength, delta_f, c=c)


def quality_factor_from_energy(energy, delta_E):
    """
    Spectral quality factor:

        Q = E / ΔE
    """
    _check_positive(energy, "energy")
    _check_positive(delta_E, "delta_E")
    return simplify(energy / delta_E)


def quality_factor_from_frequency(frequency, delta_f):
    """
    Spectral quality factor:

        Q = f / Δf
    """
    _check_positive(frequency, "frequency")
    _check_positive(delta_f, "delta_f")
    return simplify(frequency / delta_f)


def lifetime_from_linewidth_frequency(delta_f, factor: float = 0.5):
    """
    Estimate lifetime from frequency linewidth:

        τ ≈ factor / (2π Δf)
    """
    _check_positive(delta_f, "delta_f")
    return simplify(factor / (2.0 * math.pi * delta_f))


def lifetime_from_linewidth_energy(delta_E, hbar=HBAR, factor: float = 0.5):
    """
    Estimate lifetime from energy linewidth:

        τ ≈ factor * ħ / ΔE
    """
    return lifetime_from_energy_uncertainty(delta_E, hbar=hbar, factor=factor)


# ---------------------------------------------------------------------
# Convenience summaries
# ---------------------------------------------------------------------

def lifetime_broadening_summary(
    lifetime,
    photon_energy: Optional[float] = None,
    hbar=HBAR,
    h=PLANCK,
    factor: float = 0.5,
) -> Dict[str, object]:
    """
    Return a dictionary summarizing lifetime broadening.

    Parameters
    ----------
    lifetime:
        Lifetime τ in seconds.
    photon_energy:
        Optional photon energy in joule. If provided, includes Q = E/ΔE.
    factor:
        0.5 for ΔE Δt >= ħ/2. Use 1.0 for order-of-magnitude estimates.
    """
    _check_positive(lifetime, "lifetime")

    delta_E = energy_uncertainty_from_lifetime(lifetime, hbar=hbar, factor=factor)
    delta_f = frequency_uncertainty_from_energy(delta_E, h=h)
    delta_omega = delta_E / hbar

    result: Dict[str, object] = {
        "lifetime_s": lifetime,
        "delta_E_J": simplify(delta_E),
        "delta_E_eV": simplify(joule_to_ev(delta_E)),
        "delta_f_Hz": simplify(delta_f),
        "delta_omega_rad_s": simplify(delta_omega),
        "factor": factor,
    }

    if photon_energy is not None:
        _check_positive(photon_energy, "photon_energy")
        result["photon_energy_J"] = photon_energy
        result["photon_energy_eV"] = simplify(joule_to_ev(photon_energy))
        result["quality_factor"] = simplify(quality_factor_from_energy(photon_energy, delta_E))

    return result


def position_momentum_summary(delta_x, mass=ELECTRON_MASS, hbar=HBAR, factor: float = 0.5) -> Dict[str, object]:
    """
    Return a dictionary summarizing position-momentum uncertainty.

    Includes Δp and the nonrelativistic estimate Δv = Δp/m.
    """
    _check_positive(delta_x, "delta_x")
    _check_mass(mass)

    delta_p = momentum_uncertainty_from_position(delta_x, hbar=hbar, factor=factor)
    delta_v = simplify(delta_p / mass)

    return {
        "delta_x_m": delta_x,
        "mass_kg": mass,
        "delta_p_kg_m_s": simplify(delta_p),
        "delta_v_m_s": simplify(delta_v),
        "delta_v_over_c": simplify(delta_v / C),
        "factor": factor,
    }


def gamma_photon_uncertainty_from_lifetime(lifetime=1e-12, factor: float = 0.5):
    """
    Convenience helper for a common nuclear excited-state lifetime problem.

    Returns a summary of the energy uncertainty for a gamma photon emitted by
    an excited state with lifetime τ. Default τ = 1e-12 s.
    """
    return lifetime_broadening_summary(lifetime, factor=factor)


__all__ = [
    "PLANCK",
    "HBAR",
    "E_CHARGE",
    "ELECTRON_MASS",
    "joule_to_ev",
    "ev_to_joule",
    "uncertainty_lower_bound",
    "product_lower_bound",
    "satisfies_uncertainty",
    "energy_uncertainty_from_time",
    "time_uncertainty_from_energy",
    "energy_uncertainty_from_lifetime",
    "lifetime_from_energy_uncertainty",
    "angular_frequency_uncertainty_from_lifetime",
    "frequency_uncertainty_from_lifetime",
    "frequency_uncertainty_from_energy",
    "energy_uncertainty_from_frequency",
    "momentum_uncertainty_from_position",
    "position_uncertainty_from_momentum",
    "velocity_uncertainty_from_position",
    "position_uncertainty_from_velocity",
    "momentum_spread_from_wavelength_spread",
    "wavelength_spread_from_momentum_spread",
    "wavelength_uncertainty_from_frequency",
    "frequency_uncertainty_from_wavelength",
    "wavelength_uncertainty_from_lifetime",
    "quality_factor_from_energy",
    "quality_factor_from_frequency",
    "lifetime_from_linewidth_frequency",
    "lifetime_from_linewidth_energy",
    "lifetime_broadening_summary",
    "position_momentum_summary",
    "gamma_photon_uncertainty_from_lifetime",
]
