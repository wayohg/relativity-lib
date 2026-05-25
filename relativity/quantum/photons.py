"""
relativity.quantum.photons
==========================

Basic photon utilities for introductory quantum physics.

This module is designed to work with the rest of the ``relativity`` project:
it accepts ordinary floats/ints and, when possible, SymPy symbols.

Typical uses
------------
- Photon energy from frequency or wavelength.
- Photon momentum from energy, frequency, or wavelength.
- Frequency/wavelength conversions.
- Photon counting from total energy or optical power.
- Photon flux from intensity.
- Mass loss rate associated with electromagnetic radiation.

Conventions
-----------
SI is used by default:
    c in m/s, h in J s, wavelength in m, frequency in Hz, energy in J.

Electronvolt helpers are included for convenience.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover - SymPy is expected in this project.
    sp = None

try:
    from relativity.constants import C, PLANCK
except Exception:  # fallback values keep this module usable in isolation
    C = 299_792_458.0
    PLANCK = 6.62607015e-34

try:
    from relativity.utils import (
        is_symbolic,
        simplify,
        smart_array,
        smart_norm,
        normalize_vector,
    )
except Exception:  # fallback helpers for standalone use
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

    def smart_norm(v):
        v = smart_array(v)
        s = sum(component * component for component in v)
        if is_symbolic(s) and sp is not None:
            return sp.sqrt(s)
        return np.sqrt(s)

    def normalize_vector(v, name="vector"):
        v = smart_array(v)
        n = smart_norm(v)
        if not is_symbolic(n) and float(n) == 0.0:
            raise ValueError(f"{name} cannot be the zero vector.")
        return simplify(v / n)


ELEMENTARY_CHARGE = 1.602176634e-19  # C, exact in SI. Also 1 eV = e joule.
HBAR = PLANCK / (2.0 * np.pi)


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


def _sqrt(x):
    if is_symbolic(x) and sp is not None:
        return sp.sqrt(x)
    return np.sqrt(x)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def joule_to_ev(energy_joule):
    """Convert energy from joules to electronvolts."""
    return simplify(energy_joule / ELEMENTARY_CHARGE)


def ev_to_joule(energy_ev):
    """Convert energy from electronvolts to joules."""
    return simplify(energy_ev * ELEMENTARY_CHARGE)


def nm_to_m(wavelength_nm):
    """Convert wavelength from nanometers to meters."""
    return simplify(wavelength_nm * 1e-9)


def m_to_nm(wavelength_m):
    """Convert wavelength from meters to nanometers."""
    return simplify(wavelength_m / 1e-9)


# ---------------------------------------------------------------------------
# Frequency, wavelength, angular frequency and wave number
# ---------------------------------------------------------------------------


def frequency_from_wavelength(wavelength, c=C):
    """
    Return photon frequency from wavelength.

    Parameters
    ----------
    wavelength : float or symbolic
        Wavelength in meters.
    c : float or symbolic
        Speed of light.
    """
    _check_positive(wavelength, "wavelength")
    _check_positive(c, "c")
    return simplify(c / wavelength)


def wavelength_from_frequency(frequency, c=C):
    """Return photon wavelength from frequency."""
    _check_positive(frequency, "frequency")
    _check_positive(c, "c")
    return simplify(c / frequency)


def angular_frequency_from_frequency(frequency):
    """Return angular frequency omega = 2 pi f."""
    _check_positive(frequency, "frequency")
    pi = sp.pi if is_symbolic(frequency) and sp is not None else np.pi
    return simplify(2 * pi * frequency)


def frequency_from_angular_frequency(omega):
    """Return frequency f = omega / (2 pi)."""
    _check_positive(omega, "omega")
    pi = sp.pi if is_symbolic(omega) and sp is not None else np.pi
    return simplify(omega / (2 * pi))


def angular_wavenumber_from_wavelength(wavelength):
    """Return angular wave number k = 2 pi / lambda."""
    _check_positive(wavelength, "wavelength")
    pi = sp.pi if is_symbolic(wavelength) and sp is not None else np.pi
    return simplify(2 * pi / wavelength)


def wavelength_from_angular_wavenumber(k):
    """Return wavelength lambda = 2 pi / k."""
    _check_positive(k, "k")
    pi = sp.pi if is_symbolic(k) and sp is not None else np.pi
    return simplify(2 * pi / k)


def wavenumber_from_wavelength(wavelength):
    """
    Return spectroscopic wave number 1/lambda.

    This is not the angular wave number k = 2*pi/lambda.
    """
    _check_positive(wavelength, "wavelength")
    return simplify(1 / wavelength)


# ---------------------------------------------------------------------------
# Photon energy
# ---------------------------------------------------------------------------


def energy_from_frequency(frequency, h=PLANCK):
    """Return photon energy E = h f."""
    _check_positive(frequency, "frequency")
    _check_positive(h, "h")
    return simplify(h * frequency)


def frequency_from_energy(energy, h=PLANCK):
    """Return photon frequency f = E / h."""
    _check_positive(energy, "energy")
    _check_positive(h, "h")
    return simplify(energy / h)


def energy_from_wavelength(wavelength, h=PLANCK, c=C):
    """Return photon energy E = h c / lambda."""
    _check_positive(wavelength, "wavelength")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(h * c / wavelength)


def wavelength_from_energy(energy, h=PLANCK, c=C):
    """Return photon wavelength lambda = h c / E."""
    _check_positive(energy, "energy")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(h * c / energy)


def energy_from_angular_frequency(omega, hbar=HBAR):
    """Return photon energy E = hbar * omega."""
    _check_positive(omega, "omega")
    _check_positive(hbar, "hbar")
    return simplify(hbar * omega)


def angular_frequency_from_energy(energy, hbar=HBAR):
    """Return angular frequency omega = E / hbar."""
    _check_positive(energy, "energy")
    _check_positive(hbar, "hbar")
    return simplify(energy / hbar)


# Backwards/teaching-friendly aliases
photon_energy_from_frequency = energy_from_frequency
photon_energy_from_wavelength = energy_from_wavelength
photon_frequency_from_energy = frequency_from_energy
photon_wavelength_from_energy = wavelength_from_energy


# ---------------------------------------------------------------------------
# Photon momentum and four-momentum
# ---------------------------------------------------------------------------


def momentum_from_energy(energy, c=C):
    """Return photon momentum magnitude p = E / c."""
    _check_positive(energy, "energy")
    _check_positive(c, "c")
    return simplify(energy / c)


def energy_from_momentum(momentum, c=C):
    """Return photon energy E = p c for momentum magnitude p."""
    _check_nonnegative(momentum, "momentum")
    _check_positive(c, "c")
    return simplify(momentum * c)


def momentum_from_frequency(frequency, h=PLANCK, c=C):
    """Return photon momentum magnitude p = h f / c."""
    return simplify(energy_from_frequency(frequency, h=h) / c)


def momentum_from_wavelength(wavelength, h=PLANCK):
    """Return photon momentum magnitude p = h / lambda."""
    _check_positive(wavelength, "wavelength")
    _check_positive(h, "h")
    return simplify(h / wavelength)


def momentum_vector_from_energy(energy, direction: Sequence, c=C):
    """
    Return photon 3-momentum vector from energy and propagation direction.

    The direction vector is normalized internally and cannot be zero.
    """
    p_mag = momentum_from_energy(energy, c=c)
    direction_hat = normalize_vector(direction, name="photon direction")
    return simplify(p_mag * direction_hat)


def momentum_vector_from_wavelength(wavelength, direction: Sequence, h=PLANCK):
    """Return photon 3-momentum vector from wavelength and direction."""
    p_mag = momentum_from_wavelength(wavelength, h=h)
    direction_hat = normalize_vector(direction, name="photon direction")
    return simplify(p_mag * direction_hat)


def photon_four_momentum(
    *,
    energy=None,
    frequency=None,
    wavelength=None,
    direction: Optional[Sequence] = None,
    h=PLANCK,
    c=C,
):
    """
    Return photon four-momentum using convention (E/c, px, py, pz).

    Provide exactly one of ``energy``, ``frequency`` or ``wavelength``.
    If direction is omitted, +x is used.
    """
    provided = [energy is not None, frequency is not None, wavelength is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of energy, frequency, or wavelength.")

    if energy is None:
        if frequency is not None:
            energy = energy_from_frequency(frequency, h=h)
        else:
            energy = energy_from_wavelength(wavelength, h=h, c=c)

    if direction is None:
        direction = [1, 0, 0]

    p_vec = momentum_vector_from_energy(energy, direction=direction, c=c)
    return smart_array([energy / c, p_vec[0], p_vec[1], p_vec[2]])


def photon_invariant_mass_squared(energy, momentum, c=C):
    """
    Return m^2 from the energy-momentum relation.

    For a photon this should simplify to zero when p = E/c:
        m^2 = E^2/c^4 - p^2/c^2
    """
    _check_positive(c, "c")
    p = smart_array(momentum) if isinstance(momentum, (list, tuple, np.ndarray)) else momentum
    if isinstance(p, np.ndarray):
        p2 = sum(component * component for component in p)
    else:
        p2 = p * p
    return simplify(energy**2 / c**4 - p2 / c**2)


# ---------------------------------------------------------------------------
# Photon counting and flux
# ---------------------------------------------------------------------------


def photon_count_from_total_energy(
    total_energy,
    *,
    photon_energy=None,
    frequency=None,
    wavelength=None,
    h=PLANCK,
    c=C,
):
    """
    Return the number of photons in a given total energy.

    Provide one of ``photon_energy``, ``frequency`` or ``wavelength``.
    """
    _check_nonnegative(total_energy, "total_energy")

    provided = [photon_energy is not None, frequency is not None, wavelength is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of photon_energy, frequency, or wavelength.")

    if photon_energy is None:
        if frequency is not None:
            photon_energy = energy_from_frequency(frequency, h=h)
        else:
            photon_energy = energy_from_wavelength(wavelength, h=h, c=c)

    return simplify(total_energy / photon_energy)


def photon_rate_from_power(
    power,
    *,
    photon_energy=None,
    frequency=None,
    wavelength=None,
    h=PLANCK,
    c=C,
):
    """
    Return photons per second for a given optical power.
    """
    _check_nonnegative(power, "power")
    return photon_count_from_total_energy(
        power,
        photon_energy=photon_energy,
        frequency=frequency,
        wavelength=wavelength,
        h=h,
        c=c,
    )


def photon_flux_from_intensity(
    intensity,
    *,
    photon_energy=None,
    frequency=None,
    wavelength=None,
    h=PLANCK,
    c=C,
):
    """
    Return photon flux in photons/(m^2 s) from intensity in W/m^2.
    """
    _check_nonnegative(intensity, "intensity")
    return photon_rate_from_power(
        intensity,
        photon_energy=photon_energy,
        frequency=frequency,
        wavelength=wavelength,
        h=h,
        c=c,
    )


def total_energy_from_photon_count(
    photon_count,
    *,
    photon_energy=None,
    frequency=None,
    wavelength=None,
    h=PLANCK,
    c=C,
):
    """
    Return total energy carried by N identical photons.
    """
    _check_nonnegative(photon_count, "photon_count")

    provided = [photon_energy is not None, frequency is not None, wavelength is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of photon_energy, frequency, or wavelength.")

    if photon_energy is None:
        if frequency is not None:
            photon_energy = energy_from_frequency(frequency, h=h)
        else:
            photon_energy = energy_from_wavelength(wavelength, h=h, c=c)

    return simplify(photon_count * photon_energy)


# ---------------------------------------------------------------------------
# Radiation energy and equivalent mass
# ---------------------------------------------------------------------------


def mass_equivalent_from_energy(energy, c=C):
    """Return mass equivalent m = E/c^2."""
    _check_nonnegative(energy, "energy")
    _check_positive(c, "c")
    return simplify(energy / c**2)


def energy_from_mass_equivalent(mass, c=C):
    """Return energy equivalent E = m c^2."""
    _check_nonnegative(mass, "mass")
    _check_positive(c, "c")
    return simplify(mass * c**2)


def radiation_mass_loss_rate(power, c=C):
    """
    Return mass loss rate dm/dt = P/c^2 for emitted radiation power P.
    """
    _check_nonnegative(power, "power")
    _check_positive(c, "c")
    return simplify(power / c**2)


# ---------------------------------------------------------------------------
# Radiation pressure helpers
# ---------------------------------------------------------------------------


def radiation_pressure_absorbed(intensity, c=C):
    """
    Radiation pressure on a perfectly absorbing surface at normal incidence.

    P_rad = I/c
    """
    _check_nonnegative(intensity, "intensity")
    _check_positive(c, "c")
    return simplify(intensity / c)


def radiation_pressure_reflected(intensity, c=C):
    """
    Radiation pressure on a perfectly reflecting surface at normal incidence.

    P_rad = 2I/c
    """
    _check_nonnegative(intensity, "intensity")
    _check_positive(c, "c")
    return simplify(2 * intensity / c)


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------


def photon_summary_from_wavelength(wavelength, h=PLANCK, c=C, energy_unit: str = "J"):
    """
    Return a dictionary with common photon quantities from wavelength.

    Parameters
    ----------
    wavelength : float or symbolic
        Wavelength in meters.
    energy_unit : {'J', 'eV'}
        Energy unit used in the returned dictionary.
    """
    frequency = frequency_from_wavelength(wavelength, c=c)
    energy_j = energy_from_wavelength(wavelength, h=h, c=c)
    momentum = momentum_from_wavelength(wavelength, h=h)
    omega = angular_frequency_from_frequency(frequency)
    k = angular_wavenumber_from_wavelength(wavelength)

    if energy_unit.lower() == "ev":
        energy = joule_to_ev(energy_j)
        unit = "eV"
    elif energy_unit.lower() == "j":
        energy = energy_j
        unit = "J"
    else:
        raise ValueError("energy_unit must be 'J' or 'eV'.")

    return {
        "wavelength_m": wavelength,
        "frequency_hz": frequency,
        f"energy_{unit}": energy,
        "momentum_kg_m_s": momentum,
        "angular_frequency_rad_s": omega,
        "angular_wavenumber_rad_m": k,
    }


__all__ = [
    "ELEMENTARY_CHARGE",
    "HBAR",
    "joule_to_ev",
    "ev_to_joule",
    "nm_to_m",
    "m_to_nm",
    "frequency_from_wavelength",
    "wavelength_from_frequency",
    "angular_frequency_from_frequency",
    "frequency_from_angular_frequency",
    "angular_wavenumber_from_wavelength",
    "wavelength_from_angular_wavenumber",
    "wavenumber_from_wavelength",
    "energy_from_frequency",
    "frequency_from_energy",
    "energy_from_wavelength",
    "wavelength_from_energy",
    "energy_from_angular_frequency",
    "angular_frequency_from_energy",
    "photon_energy_from_frequency",
    "photon_energy_from_wavelength",
    "photon_frequency_from_energy",
    "photon_wavelength_from_energy",
    "momentum_from_energy",
    "energy_from_momentum",
    "momentum_from_frequency",
    "momentum_from_wavelength",
    "momentum_vector_from_energy",
    "momentum_vector_from_wavelength",
    "photon_four_momentum",
    "photon_invariant_mass_squared",
    "photon_count_from_total_energy",
    "photon_rate_from_power",
    "photon_flux_from_intensity",
    "total_energy_from_photon_count",
    "mass_equivalent_from_energy",
    "energy_from_mass_equivalent",
    "radiation_mass_loss_rate",
    "radiation_pressure_absorbed",
    "radiation_pressure_reflected",
    "photon_summary_from_wavelength",
]
