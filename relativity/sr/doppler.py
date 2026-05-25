"""
relativity.sr.doppler
=====================

Relativistic Doppler, aberration and beaming utilities.

The module follows the same hybrid numeric/symbolic style used by the rest
of the package: floats/NumPy arrays work normally, and SymPy expressions are
kept symbolic when possible.

Conventions
-----------
The main Doppler factor is

    D = nu_obs / nu_emit = 1 / [gamma * (1 - beta cos(theta))]

where theta is the angle, measured in the observer/lab frame, between:

    1. the source velocity vector, and
    2. the photon propagation direction from the source to the observer.

Therefore:

    theta = 0      source moving toward observer     -> blueshift, D > 1
    theta = pi     source moving away from observer  -> redshift,  D < 1
    theta = pi/2   transverse Doppler shift          -> D = 1/gamma

Equivalently, for vectors:

    D = 1 / [gamma * (1 - dot(v, n)/c)]

where n is a unit vector pointing in the photon propagation direction.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from relativity.constants import C
from relativity.utils import (
    smart_array,
    smart_dot,
    smart_norm,
    smart_sqrt,
    smart_cos,
    is_symbolic,
    simplify,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _as_vector3(v):
    """Return *v* as a 3-vector."""
    arr = smart_array(v)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _check_beta(beta_value, allow_equal: bool = False):
    """Validate |beta| < 1 numerically. Symbolic beta is left unconstrained."""
    if is_symbolic(beta_value):
        return

    b = float(abs(beta_value))
    if allow_equal:
        bad = b > 1.0
        msg = "|beta| cannot exceed 1."
    else:
        bad = b >= 1.0
        msg = "Massive-source beta must satisfy |beta| < 1."

    if bad:
        raise ValueError(f"{msg} Got beta={beta_value:g}.")


def _gamma_from_beta(beta_value):
    _check_beta(beta_value)
    return simplify(1 / smart_sqrt(1 - beta_value**2))


def _cos(theta):
    return smart_cos(theta)


def _acos(x):
    return sp.acos(x) if is_symbolic(x) else np.arccos(x)


def _normalize_direction(n):
    n = _as_vector3(n)
    norm = smart_norm(n)

    if not is_symbolic(norm) and float(norm) == 0.0:
        raise ValueError("Direction vector cannot be zero.")

    return simplify(n / norm)


# ============================================================
# BASIC DOPPLER FACTORS
# ============================================================

def doppler_factor(beta_value, theta=0, cos_theta=None):
    """
    Relativistic Doppler factor D = nu_obs / nu_emit.

    Parameters
    ----------
    beta_value:
        Dimensionless speed beta = v/c. Use beta_value >= 0 together with
        theta to describe approach/recession.
    theta:
        Angle between source velocity and photon direction, measured in the
        observer/lab frame. Ignored when cos_theta is provided.
    cos_theta:
        Optional direct value of cos(theta). Useful for symbolic work.

    Returns
    -------
    D:
        Frequency ratio nu_obs / nu_emit.
    """
    _check_beta(beta_value)
    g = _gamma_from_beta(beta_value)
    mu = cos_theta if cos_theta is not None else _cos(theta)
    return simplify(1 / (g * (1 - beta_value * mu)))


def doppler_factor_from_velocity(v, photon_direction, c=C):
    """
    Doppler factor from a velocity vector and photon direction.

    Parameters
    ----------
    v:
        Source velocity vector in the observer/lab frame.
    photon_direction:
        Direction of photon propagation from source to observer. It does not
        need to be normalized.
    c:
        Speed of light.

    Returns
    -------
    D:
        Frequency ratio nu_obs / nu_emit.
    """
    v = _as_vector3(v)
    n = _normalize_direction(photon_direction)

    beta_vec = simplify(v / c)
    beta2 = simplify(smart_dot(beta_vec, beta_vec))

    if not is_symbolic(beta2) and beta2 >= 1:
        raise ValueError("Massive-source speed must satisfy |v| < c.")

    g = simplify(1 / smart_sqrt(1 - beta2))
    beta_parallel = simplify(smart_dot(beta_vec, n))

    return simplify(1 / (g * (1 - beta_parallel)))


def longitudinal_doppler_factor(beta_value, approaching: bool = False):
    """
    Longitudinal Doppler factor.

    By default this assumes recession, so beta_value > 0 gives redshift:

        D = sqrt((1 - beta)/(1 + beta))

    With approaching=True:

        D = sqrt((1 + beta)/(1 - beta))
    """
    _check_beta(beta_value)

    if approaching:
        return simplify(smart_sqrt((1 + beta_value) / (1 - beta_value)))

    return simplify(smart_sqrt((1 - beta_value) / (1 + beta_value)))


def transverse_doppler_factor(beta_value):
    """
    Transverse Doppler factor for theta = pi/2.

    D = 1/gamma = sqrt(1 - beta^2)
    """
    _check_beta(beta_value)
    return simplify(smart_sqrt(1 - beta_value**2))


# ============================================================
# FREQUENCY, WAVELENGTH, ENERGY
# ============================================================

def observed_frequency(f_emit, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """Observed frequency f_obs = D f_emit."""
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(D * f_emit)


def emitted_frequency(f_obs, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """Emitted/rest-frame frequency f_emit = f_obs / D."""
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(f_obs / D)


def observed_wavelength(lambda_emit, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """
    Observed wavelength lambda_obs = lambda_emit / D.

    This assumes the same light speed c for both frames.
    """
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(lambda_emit / D)


def emitted_wavelength(lambda_obs, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """Emitted/rest-frame wavelength lambda_emit = D lambda_obs."""
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(D * lambda_obs)


def observed_photon_energy(E_emit, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """Observed photon energy E_obs = D E_emit."""
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(D * E_emit)


def emitted_photon_energy(E_obs, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """Emitted/rest-frame photon energy E_emit = E_obs / D."""
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(E_obs / D)


# ============================================================
# REDSHIFT
# ============================================================

def redshift_from_frequencies(f_emit, f_obs):
    """Redshift z = f_emit/f_obs - 1."""
    return simplify(f_emit / f_obs - 1)


def redshift_from_wavelengths(lambda_emit, lambda_obs):
    """Redshift z = lambda_obs/lambda_emit - 1."""
    return simplify(lambda_obs / lambda_emit - 1)


def redshift_from_doppler(doppler):
    """Redshift z = 1/D - 1."""
    return simplify(1 / doppler - 1)


def redshift_from_beta(beta_value, theta=0, cos_theta=None):
    """Redshift corresponding to beta and viewing angle."""
    return redshift_from_doppler(doppler_factor(beta_value, theta, cos_theta))


def beta_from_longitudinal_redshift(z):
    """
    Signed line-of-sight beta from longitudinal relativistic redshift.

    Convention
    ----------
    beta > 0 means recession, beta < 0 means approach.

    Formula
    -------
    beta = ((1 + z)^2 - 1) / ((1 + z)^2 + 1)
    """
    zp1 = 1 + z
    return simplify((zp1**2 - 1) / (zp1**2 + 1))


def velocity_from_longitudinal_redshift(z, c=C):
    """Signed line-of-sight velocity from longitudinal redshift."""
    return simplify(c * beta_from_longitudinal_redshift(z))


# ============================================================
# ABERRATION OF LIGHT
# ============================================================

def aberration_cos_to_comoving(cos_theta, beta_value):
    """
    Transform lab-frame photon angle to the source comoving frame.

    cos(theta') = [cos(theta) - beta] / [1 - beta cos(theta)]
    """
    _check_beta(beta_value)
    return simplify((cos_theta - beta_value) / (1 - beta_value * cos_theta))


def aberration_cos_to_lab(cos_theta_prime, beta_value):
    """
    Transform source-comoving photon angle to the lab/observer frame.

    cos(theta) = [cos(theta') + beta] / [1 + beta cos(theta')]
    """
    _check_beta(beta_value)
    return simplify((cos_theta_prime + beta_value) / (1 + beta_value * cos_theta_prime))


def aberration_angle_to_comoving(theta, beta_value):
    """Return theta' from lab-frame theta."""
    return simplify(_acos(aberration_cos_to_comoving(_cos(theta), beta_value)))


def aberration_angle_to_lab(theta_prime, beta_value):
    """Return lab-frame theta from source-comoving theta'."""
    return simplify(_acos(aberration_cos_to_lab(_cos(theta_prime), beta_value)))


# ============================================================
# RELATIVISTIC BEAMING
# ============================================================

def beamed_specific_intensity(I_emit, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """
    Relativistic beaming for specific intensity.

    For frequency-specific intensity, approximately:

        I_nu,obs = D^3 I_nu,emit
    """
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(D**3 * I_emit)


def beamed_bolometric_intensity(I_emit, beta_value=None, theta=0, cos_theta=None, doppler=None):
    """
    Relativistic beaming for bolometric intensity/luminosity.

    Common approximation:

        I_obs = D^4 I_emit
    """
    D = doppler if doppler is not None else doppler_factor(beta_value, theta, cos_theta)
    return simplify(D**4 * I_emit)


# ============================================================
# TWO-WAY / RADAR DOPPLER
# ============================================================

def two_way_doppler_factor(beta_value, approaching: bool = False):
    """
    Two-way longitudinal Doppler factor, useful for radar/echo problems.

    This is the square of the one-way longitudinal factor.

    Receding by default:

        D2 = (1 - beta)/(1 + beta)

    Approaching:

        D2 = (1 + beta)/(1 - beta)
    """
    _check_beta(beta_value)

    if approaching:
        return simplify((1 + beta_value) / (1 - beta_value))

    return simplify((1 - beta_value) / (1 + beta_value))


def radar_echo_frequency(f_emit, beta_value, approaching: bool = False):
    """Observed returned radar frequency after a two-way Doppler shift."""
    return simplify(two_way_doppler_factor(beta_value, approaching) * f_emit)


__all__ = [
    "doppler_factor",
    "doppler_factor_from_velocity",
    "longitudinal_doppler_factor",
    "transverse_doppler_factor",
    "observed_frequency",
    "emitted_frequency",
    "observed_wavelength",
    "emitted_wavelength",
    "observed_photon_energy",
    "emitted_photon_energy",
    "redshift_from_frequencies",
    "redshift_from_wavelengths",
    "redshift_from_doppler",
    "redshift_from_beta",
    "beta_from_longitudinal_redshift",
    "velocity_from_longitudinal_redshift",
    "aberration_cos_to_comoving",
    "aberration_cos_to_lab",
    "aberration_angle_to_comoving",
    "aberration_angle_to_lab",
    "beamed_specific_intensity",
    "beamed_bolometric_intensity",
    "two_way_doppler_factor",
    "radar_echo_frequency",
]
