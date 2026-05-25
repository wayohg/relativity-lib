"""
relativity.quantum.de_broglie
=============================

Utilities for matter waves and the de Broglie relations.

This module complements ``relativity.quantum.photons`` and the special
relativity modules already present in the project.  It supports ordinary
numeric values and, when possible, SymPy symbols.

Conventions
-----------
SI units are used by default:

- mass: kg
- velocity: m/s
- momentum: kg m/s
- wavelength: m
- energy: J
- frequency: Hz

Core equations
--------------
    lambda = h / p
    p = gamma m v
    E^2 = (pc)^2 + (mc^2)^2
    K = (gamma - 1) m c^2

For low speeds, the non-relativistic approximation is also provided:

    p ≈ m v
    lambda ≈ h / (m v)
    K ≈ p^2 / (2m)
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover - SymPy is expected in this project.
    sp = None


# ---------------------------------------------------------------------------
# Constants with safe fallbacks
# ---------------------------------------------------------------------------

try:
    from relativity import constants as _constants
except Exception:  # pragma: no cover
    _constants = None


def _const(*names: str, default: float) -> float:
    """Read a constant from ``relativity.constants`` using possible names."""
    if _constants is not None:
        for name in names:
            if hasattr(_constants, name):
                return getattr(_constants, name)
    return default


C = _const("C", "c", "SPEED_OF_LIGHT", default=299_792_458.0)
PLANCK = _const("PLANCK", "H", "h", "PLANCK_CONSTANT", default=6.62607015e-34)
HBAR = _const("HBAR", "hbar", "REDUCED_PLANCK", default=PLANCK / (2.0 * np.pi))
ELEMENTARY_CHARGE = _const("ELEMENTARY_CHARGE", "E_CHARGE", "Q_E", default=1.602176634e-19)
ELECTRON_MASS = _const("ELECTRON_MASS", "M_E", "MASS_ELECTRON", default=9.1093837015e-31)
PROTON_MASS = _const("PROTON_MASS", "M_P", "MASS_PROTON", default=1.67262192369e-27)
NEUTRON_MASS = _const("NEUTRON_MASS", "M_N", "MASS_NEUTRON", default=1.67492749804e-27)


# ---------------------------------------------------------------------------
# Hybrid helpers with safe fallbacks
# ---------------------------------------------------------------------------

try:
    from relativity.utils import (
        is_symbolic,
        simplify,
        smart_array,
        smart_dot,
        smart_norm,
        smart_sqrt,
        normalize_vector,
    )
except Exception:  # fallback helpers keep the file useful in isolation
    def is_symbolic(x) -> bool:
        if sp is None:
            return False
        try:
            if isinstance(x, np.ndarray):
                return any(is_symbolic(item) for item in x.flat)
            if isinstance(x, (list, tuple)):
                return any(is_symbolic(item) for item in x)
            return bool(getattr(x, "free_symbols", False))
        except Exception:
            return False

    def simplify(x):
        if sp is not None and is_symbolic(x):
            if isinstance(x, np.ndarray):
                return np.array([sp.simplify(item) for item in x], dtype=object)
            return sp.simplify(x)
        return x

    def smart_array(x):
        dtype = object if is_symbolic(x) else float
        return np.array(x, dtype=dtype)

    def smart_dot(a, b):
        a = smart_array(a)
        b = smart_array(b)
        return sum(ai * bi for ai, bi in zip(a, b))

    def smart_sqrt(x):
        if is_symbolic(x) and sp is not None:
            return sp.sqrt(x)
        return np.sqrt(x)

    def smart_norm(v):
        v = smart_array(v)
        return smart_sqrt(smart_dot(v, v))

    def normalize_vector(v, name="vector"):
        v = smart_array(v)
        n = smart_norm(v)
        if not is_symbolic(n) and float(n) == 0.0:
            raise ValueError(f"{name} cannot be the zero vector.")
        return simplify(v / n)

try:
    from relativity.quantum.photons import joule_to_ev, ev_to_joule
except Exception:  # fallback exact eV conversion
    def joule_to_ev(energy_joule):
        return simplify(energy_joule / ELEMENTARY_CHARGE)

    def ev_to_joule(energy_ev):
        return simplify(energy_ev * ELEMENTARY_CHARGE)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_numeric_scalar(x) -> bool:
    """Return True when x can be safely compared as a real scalar."""
    if is_symbolic(x):
        return False
    try:
        float(x)
        return True
    except Exception:
        return False


def _check_positive(x, name: str) -> None:
    """Validate that a numeric scalar is strictly positive."""
    if _is_numeric_scalar(x) and float(x) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _check_nonnegative(x, name: str) -> None:
    """Validate that a numeric scalar is non-negative."""
    if _is_numeric_scalar(x) and float(x) < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def _check_subluminal_speed(speed, c=C, allow_light: bool = False) -> None:
    """Validate a numeric speed for a massive particle."""
    if is_symbolic(speed) or is_symbolic(c):
        return
    speed_value = float(speed)
    c_value = float(c)
    if speed_value < 0.0:
        raise ValueError("speed must be non-negative.")
    if allow_light:
        if speed_value > c_value:
            raise ValueError("speed cannot exceed c.")
    elif speed_value >= c_value:
        raise ValueError("massive-particle speed must satisfy speed < c.")


def _as_speed(velocity) -> object:
    """Return speed from a scalar speed or vector velocity."""
    if isinstance(velocity, (list, tuple, np.ndarray)):
        return smart_norm(velocity)
    return velocity


def _maybe_abs_momentum(momentum):
    """Use magnitude for a vector momentum and absolute value for numeric scalars."""
    if isinstance(momentum, (list, tuple, np.ndarray)):
        return smart_norm(momentum)
    if is_symbolic(momentum):
        return momentum
    return abs(float(momentum))


# ---------------------------------------------------------------------------
# Core de Broglie relations
# ---------------------------------------------------------------------------


def wavelength_from_momentum(momentum, h=PLANCK):
    """
    Return de Broglie wavelength from momentum magnitude or vector.

    Parameters
    ----------
    momentum : scalar or vector
        Momentum in kg m/s.  If a vector is given, its norm is used.
    h : float or symbolic
        Planck constant.
    """
    p = _maybe_abs_momentum(momentum)
    _check_positive(p, "momentum magnitude")
    return simplify(h / p)


def momentum_from_wavelength(wavelength, h=PLANCK):
    """Return momentum magnitude p = h/lambda."""
    _check_positive(wavelength, "wavelength")
    return simplify(h / wavelength)


def angular_wavenumber_from_wavelength(wavelength):
    """Return angular wave number k = 2 pi / lambda."""
    _check_positive(wavelength, "wavelength")
    pi = sp.pi if is_symbolic(wavelength) and sp is not None else np.pi
    return simplify(2 * pi / wavelength)


def wavelength_from_angular_wavenumber(k):
    """Return wavelength lambda = 2 pi / k."""
    _check_positive(k, "angular wave number")
    pi = sp.pi if is_symbolic(k) and sp is not None else np.pi
    return simplify(2 * pi / k)


def momentum_from_angular_wavenumber(k, hbar=HBAR):
    """Return momentum p = hbar k."""
    _check_nonnegative(k, "angular wave number")
    return simplify(hbar * k)


def angular_wavenumber_from_momentum(momentum, hbar=HBAR):
    """Return angular wave number k = p/hbar."""
    p = _maybe_abs_momentum(momentum)
    _check_nonnegative(p, "momentum magnitude")
    return simplify(p / hbar)


def gamma_from_speed(speed, c=C):
    """Return Lorentz factor gamma = 1/sqrt(1 - v^2/c^2)."""
    speed = _as_speed(speed)
    _check_subluminal_speed(speed, c=c)
    return simplify(1 / smart_sqrt(1 - speed**2 / c**2))


def beta_from_speed(speed, c=C):
    """Return beta = v/c from a scalar speed or velocity vector."""
    speed = _as_speed(speed)
    _check_subluminal_speed(speed, c=c, allow_light=True)
    return simplify(speed / c)


def relativistic_momentum(mass, velocity, c=C):
    """
    Return relativistic momentum vector or scalar.

    For a scalar velocity, returns scalar signed momentum gamma m v.
    For a vector velocity, returns a vector gamma m vec(v).
    """
    _check_positive(mass, "mass")
    speed = _as_speed(velocity)
    _check_subluminal_speed(speed, c=c)
    g = gamma_from_speed(speed, c=c)

    if isinstance(velocity, (list, tuple, np.ndarray)):
        v = smart_array(velocity)
        return simplify(g * mass * v)

    return simplify(g * mass * velocity)


def nonrelativistic_momentum(mass, velocity):
    """Return classical momentum p = m v, scalar or vector."""
    _check_positive(mass, "mass")
    if isinstance(velocity, (list, tuple, np.ndarray)):
        return simplify(mass * smart_array(velocity))
    return simplify(mass * velocity)


def wavelength_from_mass_velocity(mass, velocity, h=PLANCK, c=C, relativistic: bool = True):
    """
    Return de Broglie wavelength for a massive particle from mass and velocity.

    If ``relativistic=True`` then p = gamma m v is used.  Otherwise, p = m v.
    """
    _check_positive(mass, "mass")
    speed = _as_speed(velocity)
    _check_positive(speed, "speed")

    p = relativistic_momentum(mass, velocity, c=c) if relativistic else nonrelativistic_momentum(mass, velocity)
    return wavelength_from_momentum(p, h=h)


def wavelength_from_speed(mass, speed, h=PLANCK, c=C, relativistic: bool = True):
    """Alias for ``wavelength_from_mass_velocity`` with a scalar speed."""
    return wavelength_from_mass_velocity(mass, speed, h=h, c=c, relativistic=relativistic)


def electron_wavelength_from_speed(speed, h=PLANCK, c=C, relativistic: bool = True):
    """Return the electron de Broglie wavelength at a given speed."""
    return wavelength_from_speed(ELECTRON_MASS, speed, h=h, c=c, relativistic=relativistic)


def proton_wavelength_from_speed(speed, h=PLANCK, c=C, relativistic: bool = True):
    """Return the proton de Broglie wavelength at a given speed."""
    return wavelength_from_speed(PROTON_MASS, speed, h=h, c=c, relativistic=relativistic)


def neutron_wavelength_from_speed(speed, h=PLANCK, c=C, relativistic: bool = True):
    """Return the neutron de Broglie wavelength at a given speed."""
    return wavelength_from_speed(NEUTRON_MASS, speed, h=h, c=c, relativistic=relativistic)


# ---------------------------------------------------------------------------
# Relations using kinetic energy
# ---------------------------------------------------------------------------


def rest_energy(mass, c=C):
    """Return rest energy E0 = mc^2."""
    _check_positive(mass, "mass")
    return simplify(mass * c**2)


def total_energy_from_momentum(mass, momentum, c=C):
    """Return total relativistic energy E = sqrt((pc)^2 + (mc^2)^2)."""
    _check_positive(mass, "mass")
    p = _maybe_abs_momentum(momentum)
    _check_nonnegative(p, "momentum magnitude")
    return simplify(smart_sqrt((p * c)**2 + (mass * c**2)**2))


def kinetic_energy_from_momentum(mass, momentum, c=C, relativistic: bool = True):
    """
    Return kinetic energy from momentum.

    Relativistic: K = sqrt((pc)^2 + (mc^2)^2) - mc^2.
    Non-relativistic: K = p^2/(2m).
    """
    _check_positive(mass, "mass")
    p = _maybe_abs_momentum(momentum)
    _check_nonnegative(p, "momentum magnitude")

    if relativistic:
        return simplify(total_energy_from_momentum(mass, p, c=c) - mass * c**2)

    return simplify(p**2 / (2 * mass))


def momentum_from_kinetic_energy(mass, kinetic_energy, c=C, relativistic: bool = True):
    """
    Return momentum magnitude from kinetic energy.

    Relativistic:
        pc = sqrt(K^2 + 2 K m c^2)
    Non-relativistic:
        p = sqrt(2 m K)
    """
    _check_positive(mass, "mass")
    _check_nonnegative(kinetic_energy, "kinetic energy")

    if relativistic:
        return simplify(smart_sqrt(kinetic_energy**2 + 2 * kinetic_energy * mass * c**2) / c)

    return simplify(smart_sqrt(2 * mass * kinetic_energy))


def wavelength_from_kinetic_energy(mass, kinetic_energy, h=PLANCK, c=C, relativistic: bool = True):
    """Return de Broglie wavelength from kinetic energy."""
    p = momentum_from_kinetic_energy(mass, kinetic_energy, c=c, relativistic=relativistic)
    return wavelength_from_momentum(p, h=h)


def kinetic_energy_from_wavelength(mass, wavelength, h=PLANCK, c=C, relativistic: bool = True):
    """Return kinetic energy associated with a de Broglie wavelength."""
    p = momentum_from_wavelength(wavelength, h=h)
    return kinetic_energy_from_momentum(mass, p, c=c, relativistic=relativistic)


def speed_from_kinetic_energy(mass, kinetic_energy, c=C, relativistic: bool = True):
    """
    Return speed from kinetic energy.

    Relativistic:
        gamma = 1 + K/(mc^2), beta = sqrt(1 - 1/gamma^2)
    Non-relativistic:
        v = sqrt(2K/m)
    """
    _check_positive(mass, "mass")
    _check_nonnegative(kinetic_energy, "kinetic energy")

    if relativistic:
        g = simplify(1 + kinetic_energy / (mass * c**2))
        return simplify(c * smart_sqrt(1 - 1 / g**2))

    return simplify(smart_sqrt(2 * kinetic_energy / mass))


def kinetic_energy_from_speed(mass, speed, c=C, relativistic: bool = True):
    """Return kinetic energy from speed."""
    _check_positive(mass, "mass")
    _check_nonnegative(speed, "speed")

    if relativistic:
        g = gamma_from_speed(speed, c=c)
        return simplify((g - 1) * mass * c**2)

    return simplify(0.5 * mass * speed**2)


def speed_from_wavelength(mass, wavelength, h=PLANCK, c=C, relativistic: bool = True):
    """Return speed associated with a de Broglie wavelength."""
    K = kinetic_energy_from_wavelength(mass, wavelength, h=h, c=c, relativistic=relativistic)
    return speed_from_kinetic_energy(mass, K, c=c, relativistic=relativistic)


# ---------------------------------------------------------------------------
# Frequency and wave quantities for matter waves
# ---------------------------------------------------------------------------


def total_energy_from_kinetic_energy(mass, kinetic_energy, c=C):
    """Return total relativistic energy E = mc^2 + K."""
    _check_positive(mass, "mass")
    _check_nonnegative(kinetic_energy, "kinetic energy")
    return simplify(mass * c**2 + kinetic_energy)


def matter_wave_frequency_from_total_energy(total_energy, h=PLANCK):
    """Return de Broglie/Planck frequency f = E/h from total energy."""
    _check_positive(total_energy, "total energy")
    return simplify(total_energy / h)


def matter_wave_angular_frequency_from_total_energy(total_energy, hbar=HBAR):
    """Return angular frequency omega = E/hbar from total energy."""
    _check_positive(total_energy, "total energy")
    return simplify(total_energy / hbar)


def matter_wave_frequency_from_speed(mass, speed, h=PLANCK, c=C):
    """Return matter-wave frequency using total relativistic energy."""
    g = gamma_from_speed(speed, c=c)
    E = g * mass * c**2
    return simplify(E / h)


def phase_velocity(speed, c=C):
    """
    Return phase velocity of a free relativistic matter wave.

    For massive particles, v_phase = c^2 / v, which may exceed c; this does
    not transmit information.
    """
    _check_positive(speed, "speed")
    _check_subluminal_speed(speed, c=c)
    return simplify(c**2 / speed)


def group_velocity(speed):
    """Return group velocity of a free matter wave; equal to particle speed."""
    _check_nonnegative(speed, "speed")
    return speed


# ---------------------------------------------------------------------------
# Vectors and summaries
# ---------------------------------------------------------------------------


def momentum_vector_from_wavelength(wavelength, direction: Sequence, h=PLANCK):
    """Return momentum vector with magnitude h/lambda along ``direction``."""
    p = momentum_from_wavelength(wavelength, h=h)
    direction_hat = normalize_vector(direction, name="de Broglie direction")
    return simplify(p * direction_hat)


def wavelength_from_momentum_vector(momentum_vector, h=PLANCK):
    """Return wavelength from a momentum vector."""
    return wavelength_from_momentum(momentum_vector, h=h)


def beta_from_wavelength(mass, wavelength, h=PLANCK, c=C):
    """
    Return beta = v/c associated with a de Broglie wavelength.

    Uses p = h/lambda and beta = pc/E.
    """
    _check_positive(mass, "mass")
    p = momentum_from_wavelength(wavelength, h=h)
    E = total_energy_from_momentum(mass, p, c=c)
    return simplify(p * c / E)


def speed_from_momentum(mass, momentum, c=C):
    """Return speed from momentum using beta = pc/E."""
    _check_positive(mass, "mass")
    p = _maybe_abs_momentum(momentum)
    E = total_energy_from_momentum(mass, p, c=c)
    return simplify(p * c**2 / E)


def summary_from_speed(mass, speed, h=PLANCK, c=C, energy_unit: str = "J") -> Mapping[str, object]:
    """Return a dictionary summarizing de Broglie properties from speed."""
    _check_positive(mass, "mass")
    _check_positive(speed, "speed")

    beta = beta_from_speed(speed, c=c)
    gamma = gamma_from_speed(speed, c=c)
    p_rel = relativistic_momentum(mass, speed, c=c)
    p_nonrel = nonrelativistic_momentum(mass, speed)
    lambda_rel = wavelength_from_momentum(p_rel, h=h)
    lambda_nonrel = wavelength_from_momentum(p_nonrel, h=h)
    K_rel = kinetic_energy_from_speed(mass, speed, c=c, relativistic=True)
    K_nonrel = kinetic_energy_from_speed(mass, speed, c=c, relativistic=False)
    E_total = rest_energy(mass, c=c) + K_rel
    f = matter_wave_frequency_from_total_energy(E_total, h=h)

    result = {
        "mass_kg": mass,
        "speed_m_per_s": speed,
        "beta": beta,
        "gamma": gamma,
        "momentum_relativistic_kg_m_per_s": p_rel,
        "momentum_nonrelativistic_kg_m_per_s": p_nonrel,
        "wavelength_relativistic_m": lambda_rel,
        "wavelength_nonrelativistic_m": lambda_nonrel,
        "kinetic_energy_relativistic_J": K_rel,
        "kinetic_energy_nonrelativistic_J": K_nonrel,
        "total_energy_J": E_total,
        "matter_wave_frequency_Hz": f,
        "phase_velocity_m_per_s": phase_velocity(speed, c=c),
        "group_velocity_m_per_s": group_velocity(speed),
    }

    if energy_unit.lower() == "ev":
        result.update(
            {
                "kinetic_energy_relativistic_eV": joule_to_ev(K_rel),
                "kinetic_energy_nonrelativistic_eV": joule_to_ev(K_nonrel),
                "total_energy_eV": joule_to_ev(E_total),
            }
        )

    return result


def summary_from_kinetic_energy(
    mass,
    kinetic_energy,
    h=PLANCK,
    c=C,
    relativistic: bool = True,
    energy_unit: str = "J",
) -> Mapping[str, object]:
    """Return a dictionary summarizing de Broglie properties from kinetic energy."""
    _check_positive(mass, "mass")
    _check_nonnegative(kinetic_energy, "kinetic energy")

    p = momentum_from_kinetic_energy(mass, kinetic_energy, c=c, relativistic=relativistic)
    lam = wavelength_from_momentum(p, h=h)
    speed = speed_from_kinetic_energy(mass, kinetic_energy, c=c, relativistic=relativistic)

    result = {
        "mass_kg": mass,
        "kinetic_energy_J": kinetic_energy,
        "relativistic": relativistic,
        "momentum_kg_m_per_s": p,
        "wavelength_m": lam,
        "speed_m_per_s": speed,
        "beta": simplify(speed / c),
    }

    if relativistic:
        result["gamma"] = simplify(1 + kinetic_energy / (mass * c**2))
        result["total_energy_J"] = simplify(mass * c**2 + kinetic_energy)
        result["matter_wave_frequency_Hz"] = matter_wave_frequency_from_total_energy(
            result["total_energy_J"], h=h
        )

    if energy_unit.lower() == "ev":
        result["kinetic_energy_eV"] = joule_to_ev(kinetic_energy)
        if "total_energy_J" in result:
            result["total_energy_eV"] = joule_to_ev(result["total_energy_J"])

    return result


def electron_summary_from_speed(speed, h=PLANCK, c=C, energy_unit: str = "eV") -> Mapping[str, object]:
    """Return de Broglie summary for an electron at a given speed."""
    return summary_from_speed(ELECTRON_MASS, speed, h=h, c=c, energy_unit=energy_unit)


def electron_summary_from_kinetic_energy(kinetic_energy, h=PLANCK, c=C, relativistic: bool = True, energy_unit: str = "eV") -> Mapping[str, object]:
    """Return de Broglie summary for an electron from kinetic energy in joules."""
    return summary_from_kinetic_energy(
        ELECTRON_MASS,
        kinetic_energy,
        h=h,
        c=c,
        relativistic=relativistic,
        energy_unit=energy_unit,
    )


def electron_summary_from_kinetic_energy_ev(kinetic_energy_ev, h=PLANCK, c=C, relativistic: bool = True) -> Mapping[str, object]:
    """Return electron de Broglie summary from kinetic energy in electronvolts."""
    return electron_summary_from_kinetic_energy(
        ev_to_joule(kinetic_energy_ev),
        h=h,
        c=c,
        relativistic=relativistic,
        energy_unit="eV",
    )


def compare_relativistic_nonrelativistic_wavelength(mass, speeds, h=PLANCK, c=C):
    """
    Compare relativistic and non-relativistic de Broglie wavelengths.

    Returns a dictionary of NumPy arrays useful for plotting.
    """
    speeds = np.asarray(speeds, dtype=float)
    if np.any(speeds <= 0):
        raise ValueError("all speeds must be positive.")
    if np.any(speeds >= c):
        raise ValueError("all speeds must satisfy speed < c.")

    lambda_rel = np.array([wavelength_from_speed(mass, v, h=h, c=c, relativistic=True) for v in speeds], dtype=float)
    lambda_nonrel = np.array([wavelength_from_speed(mass, v, h=h, c=c, relativistic=False) for v in speeds], dtype=float)

    return {
        "speed_m_per_s": speeds,
        "beta": speeds / c,
        "wavelength_relativistic_m": lambda_rel,
        "wavelength_nonrelativistic_m": lambda_nonrel,
        "relative_difference": (lambda_nonrel - lambda_rel) / lambda_rel,
    }


__all__ = [
    "C",
    "PLANCK",
    "HBAR",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "NEUTRON_MASS",
    "wavelength_from_momentum",
    "momentum_from_wavelength",
    "angular_wavenumber_from_wavelength",
    "wavelength_from_angular_wavenumber",
    "momentum_from_angular_wavenumber",
    "angular_wavenumber_from_momentum",
    "gamma_from_speed",
    "beta_from_speed",
    "relativistic_momentum",
    "nonrelativistic_momentum",
    "wavelength_from_mass_velocity",
    "wavelength_from_speed",
    "electron_wavelength_from_speed",
    "proton_wavelength_from_speed",
    "neutron_wavelength_from_speed",
    "rest_energy",
    "total_energy_from_momentum",
    "kinetic_energy_from_momentum",
    "momentum_from_kinetic_energy",
    "wavelength_from_kinetic_energy",
    "kinetic_energy_from_wavelength",
    "speed_from_kinetic_energy",
    "kinetic_energy_from_speed",
    "speed_from_wavelength",
    "total_energy_from_kinetic_energy",
    "matter_wave_frequency_from_total_energy",
    "matter_wave_angular_frequency_from_total_energy",
    "matter_wave_frequency_from_speed",
    "phase_velocity",
    "group_velocity",
    "momentum_vector_from_wavelength",
    "wavelength_from_momentum_vector",
    "beta_from_wavelength",
    "speed_from_momentum",
    "summary_from_speed",
    "summary_from_kinetic_energy",
    "electron_summary_from_speed",
    "electron_summary_from_kinetic_energy",
    "electron_summary_from_kinetic_energy_ev",
    "compare_relativistic_nonrelativistic_wavelength",
]
