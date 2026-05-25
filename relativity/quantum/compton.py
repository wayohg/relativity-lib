"""
relativity.quantum.compton
==========================

Utilities for Compton scattering in introductory quantum physics.

This module follows the hybrid style used by the rest of the ``relativity``
project: functions accept ordinary numeric values and, when possible, SymPy
symbols. SI units are used by default.

Conventions
-----------
- wavelength: m
- frequency: Hz
- energy: J
- mass: kg
- angle: rad
- photon scattering angle ``theta`` is measured from the incident photon
  direction.
- electron recoil angle ``phi`` is measured from the incident photon direction.

Core equations
--------------
    lambda_C = h / (m c)
    Delta lambda = lambda_C (1 - cos theta)
    lambda' = lambda + Delta lambda
    E' = E / (1 + (E / (m c^2)) (1 - cos theta))
    cot(theta/2) = (1 + h nu / (m c^2)) tan(phi)
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

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
    from relativity.utils import is_symbolic, simplify, smart_array, smart_sqrt
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

    def smart_sqrt(x):
        if is_symbolic(x) and sp is not None:
            return sp.sqrt(x)
        return np.sqrt(x)

try:
    from relativity.quantum.photons import (
        ELEMENTARY_CHARGE,
        energy_from_frequency,
        energy_from_wavelength,
        frequency_from_energy,
        wavelength_from_energy,
        wavelength_from_frequency,
        frequency_from_wavelength,
        momentum_from_energy,
        momentum_from_wavelength,
        joule_to_ev,
        ev_to_joule,
        nm_to_m,
        m_to_nm,
    )
except Exception:  # fallback definitions
    ELEMENTARY_CHARGE = 1.602176634e-19

    def energy_from_frequency(frequency, h=PLANCK):
        return simplify(h * frequency)

    def energy_from_wavelength(wavelength, h=PLANCK, c=C):
        return simplify(h * c / wavelength)

    def frequency_from_energy(energy, h=PLANCK):
        return simplify(energy / h)

    def wavelength_from_energy(energy, h=PLANCK, c=C):
        return simplify(h * c / energy)

    def wavelength_from_frequency(frequency, c=C):
        return simplify(c / frequency)

    def frequency_from_wavelength(wavelength, c=C):
        return simplify(c / wavelength)

    def momentum_from_energy(energy, c=C):
        return simplify(energy / c)

    def momentum_from_wavelength(wavelength, h=PLANCK):
        return simplify(h / wavelength)

    def joule_to_ev(energy_joule):
        return simplify(energy_joule / ELEMENTARY_CHARGE)

    def ev_to_joule(energy_ev):
        return simplify(energy_ev * ELEMENTARY_CHARGE)

    def nm_to_m(wavelength_nm):
        return simplify(wavelength_nm * 1e-9)

    def m_to_nm(wavelength_m):
        return simplify(wavelength_m / 1e-9)


ELECTRON_MASS = 9.1093837139e-31  # kg, CODATA 2022 rounded.
ELECTRON_REST_ENERGY = ELECTRON_MASS * C**2


# ---------------------------------------------------------------------------
# Validation and math helpers
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


def _cos(x):
    if is_symbolic(x) and sp is not None:
        return sp.cos(x)
    return np.cos(x)


def _sin(x):
    if is_symbolic(x) and sp is not None:
        return sp.sin(x)
    return np.sin(x)


def _tan(x):
    if is_symbolic(x) and sp is not None:
        return sp.tan(x)
    return np.tan(x)


def _atan(x):
    if is_symbolic(x) and sp is not None:
        return sp.atan(x)
    return np.arctan(x)


def _acos(x):
    if is_symbolic(x) and sp is not None:
        return sp.acos(x)
    return np.arccos(x)


def _pi_like(*args):
    if any(is_symbolic(arg) for arg in args) and sp is not None:
        return sp.pi
    return np.pi


def _as_radians(angle, degrees: bool = False):
    """Convert angle to radians if ``degrees=True``."""
    if degrees:
        pi = _pi_like(angle)
        return simplify(angle * pi / 180)
    return angle


def _safe_array(values):
    """Return an array while preserving symbolic entries when needed."""
    return smart_array(values)


# ---------------------------------------------------------------------------
# Compton wavelength and wavelength shift
# ---------------------------------------------------------------------------


def compton_wavelength(mass=ELECTRON_MASS, h=PLANCK, c=C):
    """
    Return the Compton wavelength lambda_C = h / (m c).

    Parameters
    ----------
    mass : float or symbolic
        Scatterer rest mass in kg. Defaults to the electron mass.
    h : float or symbolic
        Planck constant.
    c : float or symbolic
        Speed of light.
    """
    _check_positive(mass, "mass")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(h / (mass * c))


def reduced_compton_wavelength(mass=ELECTRON_MASS, hbar=None, h=PLANCK, c=C):
    """Return the reduced Compton wavelength hbar / (m c)."""
    _check_positive(mass, "mass")
    _check_positive(c, "c")
    if hbar is None:
        pi = _pi_like(mass, h, c)
        hbar = h / (2 * pi)
    _check_positive(hbar, "hbar")
    return simplify(hbar / (mass * c))


def electron_compton_wavelength(h=PLANCK, c=C):
    """Return the electron Compton wavelength."""
    return compton_wavelength(ELECTRON_MASS, h=h, c=c)


def compton_shift(
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
    particle_mass=None,
):
    """
    Return the Compton wavelength shift Delta lambda.

    Delta lambda = lambda_C (1 - cos theta)

    Parameters
    ----------
    mass, particle_mass : float or symbolic
        Mass of the scattering particle. ``particle_mass`` is accepted as an
        alias for compatibility with examples/tests. If both are supplied,
        ``particle_mass`` takes precedence.
    """
    if particle_mass is not None:
        mass = particle_mass
    theta = _as_radians(theta, degrees=degrees)
    lam_c = compton_wavelength(mass=mass, h=h, c=c)
    return simplify(lam_c * (1 - _cos(theta)))


def scattered_wavelength(
    initial_wavelength,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """Return the scattered photon wavelength lambda' = lambda + Delta lambda."""
    _check_positive(initial_wavelength, "initial_wavelength")
    return simplify(initial_wavelength + compton_shift(theta, mass=mass, h=h, c=c, degrees=degrees))


def initial_wavelength_from_scattered(
    scattered_wavelength_value,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """Return incident wavelength lambda = lambda' - Delta lambda."""
    _check_positive(scattered_wavelength_value, "scattered_wavelength")
    result = simplify(scattered_wavelength_value - compton_shift(theta, mass=mass, h=h, c=c, degrees=degrees))
    if _is_numeric_scalar(result) and float(result) <= 0.0:
        raise ValueError("scattered_wavelength is too small for the given scattering angle.")
    return result


def max_compton_shift(mass=ELECTRON_MASS, h=PLANCK, c=C):
    """Return maximum Compton shift, obtained at theta = pi: Delta lambda_max = 2 lambda_C."""
    return simplify(2 * compton_wavelength(mass=mass, h=h, c=c))


def fractional_compton_shift(initial_wavelength, theta, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """Return Delta lambda / lambda."""
    _check_positive(initial_wavelength, "initial_wavelength")
    return simplify(compton_shift(theta, mass=mass, h=h, c=c, degrees=degrees) / initial_wavelength)


def max_fractional_compton_shift(initial_wavelength, mass=ELECTRON_MASS, h=PLANCK, c=C):
    """Return the maximum fractional shift, 2 lambda_C / lambda."""
    _check_positive(initial_wavelength, "initial_wavelength")
    return simplify(max_compton_shift(mass=mass, h=h, c=c) / initial_wavelength)


def scattering_angle_from_shift(delta_wavelength, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """
    Return photon scattering angle from a measured Compton shift.

    cos(theta) = 1 - Delta lambda / lambda_C
    """
    _check_nonnegative(delta_wavelength, "delta_wavelength")
    lam_c = compton_wavelength(mass=mass, h=h, c=c)
    arg = simplify(1 - delta_wavelength / lam_c)

    if _is_numeric_scalar(arg) and not (-1.0 <= float(arg) <= 1.0):
        raise ValueError("delta_wavelength must satisfy 0 <= delta <= 2 lambda_C.")

    theta = _acos(arg)
    if degrees:
        pi = _pi_like(theta)
        return simplify(theta * 180 / pi)
    return simplify(theta)


def scattering_angle_from_wavelengths(initial_wavelength, scattered_wavelength_value, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """Return photon scattering angle from lambda and lambda'."""
    _check_positive(initial_wavelength, "initial_wavelength")
    _check_positive(scattered_wavelength_value, "scattered_wavelength")
    return scattering_angle_from_shift(
        scattered_wavelength_value - initial_wavelength,
        mass=mass,
        h=h,
        c=c,
        degrees=degrees,
    )


# ---------------------------------------------------------------------------
# Energy, frequency and momentum after scattering
# ---------------------------------------------------------------------------


def scattered_photon_energy_from_energy(
    initial_energy,
    theta,
    mass=ELECTRON_MASS,
    c=C,
    degrees: bool = False,
):
    """
    Return scattered photon energy E'.

    E' = E / (1 + (E / (m c^2)) (1 - cos theta))
    """
    _check_positive(initial_energy, "initial_energy")
    _check_positive(mass, "mass")
    _check_positive(c, "c")
    theta = _as_radians(theta, degrees=degrees)
    alpha = initial_energy / (mass * c**2)
    return simplify(initial_energy / (1 + alpha * (1 - _cos(theta))))


def scattered_photon_energy_from_wavelength(
    initial_wavelength,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """Return scattered photon energy E' from incident wavelength and angle."""
    e0 = energy_from_wavelength(initial_wavelength, h=h, c=c)
    return scattered_photon_energy_from_energy(e0, theta, mass=mass, c=c, degrees=degrees)


def scattered_frequency_from_frequency(
    initial_frequency,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """Return scattered photon frequency nu'."""
    e0 = energy_from_frequency(initial_frequency, h=h)
    ep = scattered_photon_energy_from_energy(e0, theta, mass=mass, c=c, degrees=degrees)
    return frequency_from_energy(ep, h=h)


def scattered_frequency_from_wavelength(
    initial_wavelength,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """Return scattered photon frequency from incident wavelength and angle."""
    ep = scattered_photon_energy_from_wavelength(initial_wavelength, theta, mass=mass, h=h, c=c, degrees=degrees)
    return frequency_from_energy(ep, h=h)


def scattered_photon_momentum_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """Return scattered photon momentum magnitude p' = E'/c."""
    ep = scattered_photon_energy_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees)
    return momentum_from_energy(ep, c=c)


def scattered_photon_momentum_from_wavelength(initial_wavelength, theta, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """Return scattered photon momentum magnitude from incident wavelength and angle."""
    ep = scattered_photon_energy_from_wavelength(initial_wavelength, theta, mass=mass, h=h, c=c, degrees=degrees)
    return momentum_from_energy(ep, c=c)


def energy_transfer_to_recoil_particle_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """Return kinetic energy transferred to the recoiling particle, K = E - E'."""
    ep = scattered_photon_energy_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees)
    return simplify(initial_energy - ep)


def recoil_kinetic_energy_from_wavelength(initial_wavelength, theta, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """Return recoil particle kinetic energy from incident wavelength and angle."""
    e0 = energy_from_wavelength(initial_wavelength, h=h, c=c)
    return energy_transfer_to_recoil_particle_from_energy(e0, theta, mass=mass, c=c, degrees=degrees)


def recoil_total_energy_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """Return total relativistic energy of the recoiling particle."""
    return simplify(mass * c**2 + energy_transfer_to_recoil_particle_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees))


def recoil_momentum_magnitude_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """
    Return recoiling particle momentum magnitude from momentum conservation.

    Incident photon travels along +x. The scattered photon leaves at angle theta.
    """
    theta = _as_radians(theta, degrees=degrees)
    p0 = momentum_from_energy(initial_energy, c=c)
    ep = scattered_photon_energy_from_energy(initial_energy, theta, mass=mass, c=c)
    pp = momentum_from_energy(ep, c=c)
    return simplify(smart_sqrt(p0**2 + pp**2 - 2 * p0 * pp * _cos(theta)))


def recoil_momentum_vector_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """
    Return recoil particle 3-momentum vector [px, py, pz].

    Assumes incident photon along +x and scattered photon in the x-y plane.
    Momentum conservation gives:
        p_e,x = p - p' cos(theta)
        p_e,y = -p' sin(theta)
        p_e,z = 0
    """
    theta = _as_radians(theta, degrees=degrees)
    p0 = momentum_from_energy(initial_energy, c=c)
    ep = scattered_photon_energy_from_energy(initial_energy, theta, mass=mass, c=c)
    pp = momentum_from_energy(ep, c=c)
    return _safe_array([simplify(p0 - pp * _cos(theta)), simplify(-pp * _sin(theta)), 0])


def recoil_speed_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """
    Return recoil particle speed using v = p c^2 / E_total.
    """
    p = recoil_momentum_magnitude_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees)
    e_total = recoil_total_energy_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees)
    return simplify(p * c**2 / e_total)


def recoil_beta_from_energy(initial_energy, theta, mass=ELECTRON_MASS, c=C, degrees: bool = False):
    """Return recoil particle beta = v/c."""
    return simplify(recoil_speed_from_energy(initial_energy, theta, mass=mass, c=c, degrees=degrees) / c)


# ---------------------------------------------------------------------------
# Photon-electron angle relation
# ---------------------------------------------------------------------------


def recoil_angle_from_photon_angle(
    theta,
    initial_frequency=None,
    initial_energy=None,
    initial_wavelength=None,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """
    Return the recoil angle phi from the scattered photon angle theta.

    Uses the relation:
        cot(theta/2) = (1 + E/(m c^2)) tan(phi)

    Provide exactly one of ``initial_frequency``, ``initial_energy`` or
    ``initial_wavelength``.
    """
    e0 = _initial_energy_from_any(
        initial_frequency=initial_frequency,
        initial_energy=initial_energy,
        initial_wavelength=initial_wavelength,
        h=h,
        c=c,
    )
    theta = _as_radians(theta, degrees=degrees)
    alpha = e0 / (mass * c**2)
    tan_phi = simplify(1 / ((1 + alpha) * _tan(theta / 2)))
    phi = _atan(tan_phi)
    if degrees:
        pi = _pi_like(phi)
        return simplify(phi * 180 / pi)
    return simplify(phi)


def photon_angle_from_recoil_angle(
    phi,
    initial_frequency=None,
    initial_energy=None,
    initial_wavelength=None,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
):
    """
    Return photon scattering angle theta from recoil angle phi.

    From cot(theta/2) = (1 + alpha) tan(phi),
        theta = 2 atan(1 / ((1 + alpha) tan(phi)))
    """
    e0 = _initial_energy_from_any(
        initial_frequency=initial_frequency,
        initial_energy=initial_energy,
        initial_wavelength=initial_wavelength,
        h=h,
        c=c,
    )
    phi = _as_radians(phi, degrees=degrees)
    alpha = e0 / (mass * c**2)
    theta = simplify(2 * _atan(1 / ((1 + alpha) * _tan(phi))))
    if degrees:
        pi = _pi_like(theta)
        return simplify(theta * 180 / pi)
    return simplify(theta)


def angle_relation_lhs(theta, degrees: bool = False):
    """Return cot(theta/2). Useful for checking the Compton angle relation."""
    theta = _as_radians(theta, degrees=degrees)
    return simplify(1 / _tan(theta / 2))


def angle_relation_rhs(phi, initial_frequency=None, initial_energy=None, initial_wavelength=None, mass=ELECTRON_MASS, h=PLANCK, c=C, degrees: bool = False):
    """Return (1 + E/(m c^2)) tan(phi)."""
    e0 = _initial_energy_from_any(
        initial_frequency=initial_frequency,
        initial_energy=initial_energy,
        initial_wavelength=initial_wavelength,
        h=h,
        c=c,
    )
    phi = _as_radians(phi, degrees=degrees)
    return simplify((1 + e0 / (mass * c**2)) * _tan(phi))


# ---------------------------------------------------------------------------
# Convenience wrappers and summaries
# ---------------------------------------------------------------------------


def _initial_energy_from_any(*, initial_frequency=None, initial_energy=None, initial_wavelength=None, h=PLANCK, c=C):
    provided = [initial_frequency is not None, initial_energy is not None, initial_wavelength is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of initial_frequency, initial_energy, or initial_wavelength.")
    if initial_energy is not None:
        _check_positive(initial_energy, "initial_energy")
        return initial_energy
    if initial_frequency is not None:
        return energy_from_frequency(initial_frequency, h=h)
    return energy_from_wavelength(initial_wavelength, h=h, c=c)


def photon_scattering_summary(
    *,
    initial_wavelength=None,
    initial_frequency=None,
    initial_energy=None,
    theta,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
    degrees: bool = False,
) -> Mapping[str, object]:
    """
    Return a dictionary with the main quantities in Compton scattering.

    Provide exactly one of initial_wavelength, initial_frequency or initial_energy.
    """
    e0 = _initial_energy_from_any(
        initial_frequency=initial_frequency,
        initial_energy=initial_energy,
        initial_wavelength=initial_wavelength,
        h=h,
        c=c,
    )
    if initial_wavelength is None:
        initial_wavelength = wavelength_from_energy(e0, h=h, c=c)
    if initial_frequency is None:
        initial_frequency = frequency_from_energy(e0, h=h)

    theta_rad = _as_radians(theta, degrees=degrees)
    delta_lambda = compton_shift(theta_rad, mass=mass, h=h, c=c)
    lambda_prime = scattered_wavelength(initial_wavelength, theta_rad, mass=mass, h=h, c=c)
    e_prime = scattered_photon_energy_from_energy(e0, theta_rad, mass=mass, c=c)
    f_prime = frequency_from_energy(e_prime, h=h)
    k_recoil = simplify(e0 - e_prime)
    p_recoil = recoil_momentum_magnitude_from_energy(e0, theta_rad, mass=mass, c=c)
    beta_recoil = recoil_beta_from_energy(e0, theta_rad, mass=mass, c=c)
    phi = recoil_angle_from_photon_angle(theta_rad, initial_energy=e0, mass=mass, h=h, c=c)

    return {
        "theta_rad": theta_rad,
        "theta_deg": simplify(theta_rad * 180 / _pi_like(theta_rad)),
        "initial_wavelength_m": initial_wavelength,
        "initial_frequency_hz": initial_frequency,
        "initial_energy_j": e0,
        "initial_energy_ev": joule_to_ev(e0),
        "compton_wavelength_m": compton_wavelength(mass=mass, h=h, c=c),
        "delta_wavelength_m": delta_lambda,
        "delta_wavelength_nm": m_to_nm(delta_lambda),
        "scattered_wavelength_m": lambda_prime,
        "scattered_wavelength_nm": m_to_nm(lambda_prime),
        "scattered_frequency_hz": f_prime,
        "scattered_energy_j": e_prime,
        "scattered_energy_ev": joule_to_ev(e_prime),
        "recoil_kinetic_energy_j": k_recoil,
        "recoil_kinetic_energy_ev": joule_to_ev(k_recoil),
        "recoil_momentum_kg_m_s": p_recoil,
        "recoil_beta": beta_recoil,
        "recoil_angle_rad": phi,
        "recoil_angle_deg": simplify(phi * 180 / _pi_like(phi)),
    }


def max_fractional_shift_summary(initial_wavelength, mass=ELECTRON_MASS, h=PLANCK, c=C) -> Mapping[str, object]:
    """Return a compact summary for the maximum Compton shift at theta = pi."""
    _check_positive(initial_wavelength, "initial_wavelength")
    delta_max = max_compton_shift(mass=mass, h=h, c=c)
    frac = simplify(delta_max / initial_wavelength)
    return {
        "initial_wavelength_m": initial_wavelength,
        "initial_wavelength_nm": m_to_nm(initial_wavelength),
        "compton_wavelength_m": compton_wavelength(mass=mass, h=h, c=c),
        "max_delta_wavelength_m": delta_max,
        "max_delta_wavelength_nm": m_to_nm(delta_max),
        "max_fractional_shift": frac,
        "max_fractional_shift_percent": simplify(100 * frac),
    }


def sample_compton_shift(
    initial_wavelength,
    theta_min=0.0,
    theta_max=np.pi,
    num: int = 400,
    mass=ELECTRON_MASS,
    h=PLANCK,
    c=C,
):
    """
    Generate numeric arrays for plotting Compton shift versus angle.

    Returns
    -------
    theta : ndarray
        Angles in radians.
    delta_lambda : ndarray
        Wavelength shifts in meters.
    scattered_lambda : ndarray
        Scattered wavelengths in meters.
    """
    _check_positive(initial_wavelength, "initial_wavelength")
    if num < 2:
        raise ValueError("num must be at least 2.")
    theta = np.linspace(float(theta_min), float(theta_max), int(num))
    delta = compton_shift(theta, mass=mass, h=h, c=c)
    lam_prime = initial_wavelength + delta
    return theta, delta, lam_prime


# Teaching-friendly aliases
compton_delta_wavelength = compton_shift
compton_scattered_wavelength = scattered_wavelength
compton_energy_scattered = scattered_photon_energy_from_energy
compton_max_fractional_shift = max_fractional_compton_shift


__all__ = [
    "ELECTRON_MASS",
    "ELECTRON_REST_ENERGY",
    "ELEMENTARY_CHARGE",
    "joule_to_ev",
    "ev_to_joule",
    "nm_to_m",
    "m_to_nm",
    "compton_wavelength",
    "reduced_compton_wavelength",
    "electron_compton_wavelength",
    "compton_shift",
    "scattered_wavelength",
    "initial_wavelength_from_scattered",
    "max_compton_shift",
    "fractional_compton_shift",
    "max_fractional_compton_shift",
    "scattering_angle_from_shift",
    "scattering_angle_from_wavelengths",
    "scattered_photon_energy_from_energy",
    "scattered_photon_energy_from_wavelength",
    "scattered_frequency_from_frequency",
    "scattered_frequency_from_wavelength",
    "scattered_photon_momentum_from_energy",
    "scattered_photon_momentum_from_wavelength",
    "energy_transfer_to_recoil_particle_from_energy",
    "recoil_kinetic_energy_from_wavelength",
    "recoil_total_energy_from_energy",
    "recoil_momentum_magnitude_from_energy",
    "recoil_momentum_vector_from_energy",
    "recoil_speed_from_energy",
    "recoil_beta_from_energy",
    "recoil_angle_from_photon_angle",
    "photon_angle_from_recoil_angle",
    "angle_relation_lhs",
    "angle_relation_rhs",
    "photon_scattering_summary",
    "max_fractional_shift_summary",
    "sample_compton_shift",
    "compton_delta_wavelength",
    "compton_scattered_wavelength",
    "compton_energy_scattered",
    "compton_max_fractional_shift",
]
