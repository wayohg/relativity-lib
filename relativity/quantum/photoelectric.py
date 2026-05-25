"""
relativity.quantum.photoelectric
================================

Utilities for the photoelectric effect in introductory quantum physics.

The module is designed to fit the hybrid style of the ``relativity`` project:
most functions work with ordinary numeric values and, when possible, SymPy
symbols. SI units are used by default.

Conventions
-----------
- frequency: Hz
- wavelength: m
- energy/work function/kinetic energy: J by default
- stopping potential: V
- electron charge magnitude: e > 0

Core equations
--------------
    K_max = h f - phi
    K_max = h c / lambda - phi
    K_max = e V_stop
    f0 = phi / h
    lambda0 = h c / phi

where ``phi`` is the work function.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

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
    from relativity.utils import is_symbolic, simplify, smart_array
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

try:
    from relativity.quantum.photons import (
        ELEMENTARY_CHARGE,
        energy_from_frequency,
        energy_from_wavelength,
        frequency_from_wavelength,
        wavelength_from_frequency,
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

    def frequency_from_wavelength(wavelength, c=C):
        return simplify(c / wavelength)

    def wavelength_from_frequency(frequency, c=C):
        return simplify(c / frequency)

    def joule_to_ev(energy_joule):
        return simplify(energy_joule / ELEMENTARY_CHARGE)

    def ev_to_joule(energy_ev):
        return simplify(energy_ev * ELEMENTARY_CHARGE)

    def nm_to_m(wavelength_nm):
        return simplify(wavelength_nm * 1e-9)

    def m_to_nm(wavelength_m):
        return simplify(wavelength_m / 1e-9)


ELECTRON_MASS = 9.1093837139e-31  # kg, CODATA 2022 rounded.


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


def _check_wavelength_frequency_energy_units(energy_unit: str) -> str:
    unit = energy_unit.lower()
    if unit not in {"j", "ev"}:
        raise ValueError("energy_unit must be 'J' or 'eV'.")
    return unit


def _as_energy_joule(value, unit: str):
    """Convert an energy-like value to joules."""
    unit = _check_wavelength_frequency_energy_units(unit)
    if unit == "ev":
        return ev_to_joule(value)
    return value


def _from_energy_joule(value, unit: str):
    """Convert a joule energy to the requested unit."""
    unit = _check_wavelength_frequency_energy_units(unit)
    if unit == "ev":
        return joule_to_ev(value)
    return value


def _clip_numeric_zero(x):
    """
    Avoid tiny negative roundoff in quantities that should be zero.

    This helper only affects numeric scalar values close to zero. Symbolic
    expressions and arrays are returned untouched.
    """
    if _is_numeric_scalar(x) and abs(float(x)) < 1e-30:
        return 0.0
    return x


# ---------------------------------------------------------------------------
# Core photoelectric equations
# ---------------------------------------------------------------------------


def max_kinetic_energy_from_frequency(
    frequency,
    work_function,
    *,
    h=PLANCK,
    allow_negative: bool = False,
):
    """
    Return maximum photoelectron kinetic energy from incident frequency.

    Parameters
    ----------
    frequency : float or symbolic
        Incident light frequency in Hz.
    work_function : float or symbolic
        Work function in joules.
    allow_negative : bool
        If False, numeric values below threshold are clipped to 0. If True,
        the raw expression ``h*frequency - work_function`` is returned.
    """
    _check_positive(frequency, "frequency")
    _check_nonnegative(work_function, "work_function")
    _check_positive(h, "h")

    kinetic = simplify(energy_from_frequency(frequency, h=h) - work_function)

    if allow_negative or is_symbolic(kinetic):
        return simplify(kinetic)

    kinetic = _clip_numeric_zero(kinetic)
    return max(float(kinetic), 0.0)


def max_kinetic_energy_from_wavelength(
    wavelength,
    work_function,
    *,
    h=PLANCK,
    c=C,
    allow_negative: bool = False,
):
    """
    Return maximum photoelectron kinetic energy from incident wavelength.

    Wavelength is in meters and work function in joules.
    """
    _check_positive(wavelength, "wavelength")
    _check_nonnegative(work_function, "work_function")
    _check_positive(h, "h")
    _check_positive(c, "c")

    kinetic = simplify(energy_from_wavelength(wavelength, h=h, c=c) - work_function)

    if allow_negative or is_symbolic(kinetic):
        return simplify(kinetic)

    kinetic = _clip_numeric_zero(kinetic)
    return max(float(kinetic), 0.0)


def kinetic_energy_from_stopping_potential(stopping_potential, e=ELEMENTARY_CHARGE):
    """
    Return K_max = e V_stop.

    ``e`` is the positive elementary charge magnitude.
    """
    _check_nonnegative(stopping_potential, "stopping_potential")
    _check_positive(e, "e")
    return simplify(e * stopping_potential)


def stopping_potential_from_kinetic_energy(kinetic_energy, e=ELEMENTARY_CHARGE):
    """Return stopping potential V_stop = K_max / e."""
    _check_nonnegative(kinetic_energy, "kinetic_energy")
    _check_positive(e, "e")
    return simplify(kinetic_energy / e)


def stopping_potential_from_frequency(
    frequency,
    work_function,
    *,
    h=PLANCK,
    e=ELEMENTARY_CHARGE,
    allow_negative: bool = False,
):
    """Return stopping potential for a given frequency and work function."""
    kinetic = max_kinetic_energy_from_frequency(
        frequency,
        work_function,
        h=h,
        allow_negative=allow_negative,
    )
    return simplify(kinetic / e)


def stopping_potential_from_wavelength(
    wavelength,
    work_function,
    *,
    h=PLANCK,
    c=C,
    e=ELEMENTARY_CHARGE,
    allow_negative: bool = False,
):
    """Return stopping potential for a given wavelength and work function."""
    kinetic = max_kinetic_energy_from_wavelength(
        wavelength,
        work_function,
        h=h,
        c=c,
        allow_negative=allow_negative,
    )
    return simplify(kinetic / e)


def work_function_from_frequency_and_kinetic_energy(
    frequency,
    kinetic_energy,
    *,
    h=PLANCK,
):
    """Return work function phi = h f - K_max."""
    _check_positive(frequency, "frequency")
    _check_nonnegative(kinetic_energy, "kinetic_energy")
    _check_positive(h, "h")
    return simplify(energy_from_frequency(frequency, h=h) - kinetic_energy)


def work_function_from_wavelength_and_kinetic_energy(
    wavelength,
    kinetic_energy,
    *,
    h=PLANCK,
    c=C,
):
    """Return work function phi = h c/lambda - K_max."""
    _check_positive(wavelength, "wavelength")
    _check_nonnegative(kinetic_energy, "kinetic_energy")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(energy_from_wavelength(wavelength, h=h, c=c) - kinetic_energy)


def work_function_from_frequency_and_stopping_potential(
    frequency,
    stopping_potential,
    *,
    h=PLANCK,
    e=ELEMENTARY_CHARGE,
):
    """Return phi = h f - e V_stop."""
    kinetic = kinetic_energy_from_stopping_potential(stopping_potential, e=e)
    return work_function_from_frequency_and_kinetic_energy(frequency, kinetic, h=h)


def work_function_from_wavelength_and_stopping_potential(
    wavelength,
    stopping_potential,
    *,
    h=PLANCK,
    c=C,
    e=ELEMENTARY_CHARGE,
):
    """Return phi = h c/lambda - e V_stop."""
    kinetic = kinetic_energy_from_stopping_potential(stopping_potential, e=e)
    return work_function_from_wavelength_and_kinetic_energy(
        wavelength,
        kinetic,
        h=h,
        c=c,
    )


# ---------------------------------------------------------------------------
# Threshold / cutoff quantities
# ---------------------------------------------------------------------------


def threshold_frequency(work_function, *, h=PLANCK):
    """Return cutoff frequency f0 = phi / h."""
    _check_nonnegative(work_function, "work_function")
    _check_positive(h, "h")
    return simplify(work_function / h)


def cutoff_frequency(work_function, *, h=PLANCK):
    """Alias for threshold_frequency."""
    return threshold_frequency(work_function, h=h)


def threshold_wavelength(work_function, *, h=PLANCK, c=C):
    """Return cutoff wavelength lambda0 = h c / phi."""
    _check_positive(work_function, "work_function")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(h * c / work_function)


def cutoff_wavelength(work_function, *, h=PLANCK, c=C):
    """Alias for threshold_wavelength."""
    return threshold_wavelength(work_function, h=h, c=c)


def work_function_from_threshold_frequency(frequency0, *, h=PLANCK):
    """Return phi = h f0."""
    _check_positive(frequency0, "frequency0")
    _check_positive(h, "h")
    return simplify(h * frequency0)


def work_function_from_threshold_wavelength(wavelength0, *, h=PLANCK, c=C):
    """Return phi = h c / lambda0."""
    _check_positive(wavelength0, "wavelength0")
    _check_positive(h, "h")
    _check_positive(c, "c")
    return simplify(h * c / wavelength0)


# ---------------------------------------------------------------------------
# Solving from two measurements
# ---------------------------------------------------------------------------


def planck_constant_from_stopping_data(
    frequency_1,
    stopping_potential_1,
    frequency_2,
    stopping_potential_2,
    *,
    e=ELEMENTARY_CHARGE,
):
    """
    Estimate Planck's constant from two photoelectric measurements.

    From e V = h f - phi, the slope of V(f) is h/e:
        h = e (V2 - V1)/(f2 - f1)
    """
    _check_positive(frequency_1, "frequency_1")
    _check_positive(frequency_2, "frequency_2")
    _check_nonnegative(stopping_potential_1, "stopping_potential_1")
    _check_nonnegative(stopping_potential_2, "stopping_potential_2")
    _check_positive(e, "e")

    if _is_numeric_scalar(frequency_1) and _is_numeric_scalar(frequency_2):
        if float(frequency_1) == float(frequency_2):
            raise ValueError("frequency_1 and frequency_2 must be different.")

    return simplify(e * (stopping_potential_2 - stopping_potential_1) / (frequency_2 - frequency_1))


def work_function_from_two_stopping_data(
    frequency_1,
    stopping_potential_1,
    frequency_2,
    stopping_potential_2,
    *,
    e=ELEMENTARY_CHARGE,
):
    """
    Return work function from two stopping-potential measurements.

    The function first determines h from the two data points and then uses
    phi = h f - e V.
    """
    h_est = planck_constant_from_stopping_data(
        frequency_1,
        stopping_potential_1,
        frequency_2,
        stopping_potential_2,
        e=e,
    )
    return work_function_from_frequency_and_stopping_potential(
        frequency_1,
        stopping_potential_1,
        h=h_est,
        e=e,
    )


def cutoff_frequency_from_two_stopping_data(
    frequency_1,
    stopping_potential_1,
    frequency_2,
    stopping_potential_2,
):
    """
    Return cutoff frequency from two V_stop(f) measurements.

    The line is V = slope*f + intercept. At cutoff, V = 0:
        f0 = -intercept/slope
    """
    if _is_numeric_scalar(frequency_1) and _is_numeric_scalar(frequency_2):
        if float(frequency_1) == float(frequency_2):
            raise ValueError("frequency_1 and frequency_2 must be different.")

    slope = simplify((stopping_potential_2 - stopping_potential_1) / (frequency_2 - frequency_1))
    intercept = simplify(stopping_potential_1 - slope * frequency_1)
    return simplify(-intercept / slope)


def photoelectric_line_from_two_points(
    frequency_1,
    stopping_potential_1,
    frequency_2,
    stopping_potential_2,
):
    """
    Return slope and intercept of V_stop = slope*f + intercept.
    """
    if _is_numeric_scalar(frequency_1) and _is_numeric_scalar(frequency_2):
        if float(frequency_1) == float(frequency_2):
            raise ValueError("frequency_1 and frequency_2 must be different.")

    slope = simplify((stopping_potential_2 - stopping_potential_1) / (frequency_2 - frequency_1))
    intercept = simplify(stopping_potential_1 - slope * frequency_1)
    return {"slope_V_per_Hz": slope, "intercept_V": intercept}


def fit_stopping_potential_data(
    frequencies: Sequence[float],
    stopping_potentials: Sequence[float],
    *,
    e=ELEMENTARY_CHARGE,
):
    """
    Linear fit for stopping-potential data.

    Fits:
        V_stop = (h/e) f - phi/e

    Returns a dictionary with slope, intercept, h, work function, and cutoff
    frequency. This function is numeric because it uses NumPy least squares.
    """
    f = np.asarray(frequencies, dtype=float)
    V = np.asarray(stopping_potentials, dtype=float)

    if f.ndim != 1 or V.ndim != 1:
        raise ValueError("frequencies and stopping_potentials must be 1D arrays.")
    if len(f) != len(V):
        raise ValueError("frequencies and stopping_potentials must have the same length.")
    if len(f) < 2:
        raise ValueError("At least two data points are required.")
    if np.any(f <= 0):
        raise ValueError("All frequencies must be positive.")
    if np.any(V < 0):
        raise ValueError("Stopping potentials must be non-negative.")

    slope, intercept = np.polyfit(f, V, deg=1)
    h_est = e * slope
    work_function = -e * intercept
    f0 = -intercept / slope

    return {
        "slope_V_per_Hz": slope,
        "intercept_V": intercept,
        "h_J_s": h_est,
        "work_function_J": work_function,
        "work_function_joule": work_function,
        "work_function_eV": joule_to_ev(work_function),
        "work_function_ev": joule_to_ev(work_function),
        "cutoff_frequency_Hz": f0,
        "cutoff_wavelength_m": wavelength_from_frequency(f0),
    }


def fit_kinetic_energy_data(
    frequencies: Sequence[float],
    kinetic_energies: Sequence[float],
):
    """
    Linear fit for K_max(f) data.

    Fits:
        K_max = h f - phi

    Returns h, work function, and cutoff frequency. This function is numeric.
    """
    f = np.asarray(frequencies, dtype=float)
    K = np.asarray(kinetic_energies, dtype=float)

    if f.ndim != 1 or K.ndim != 1:
        raise ValueError("frequencies and kinetic_energies must be 1D arrays.")
    if len(f) != len(K):
        raise ValueError("frequencies and kinetic_energies must have the same length.")
    if len(f) < 2:
        raise ValueError("At least two data points are required.")
    if np.any(f <= 0):
        raise ValueError("All frequencies must be positive.")
    if np.any(K < 0):
        raise ValueError("Kinetic energies must be non-negative.")

    slope, intercept = np.polyfit(f, K, deg=1)
    h_est = slope
    work_function = -intercept
    f0 = -intercept / slope

    return {
        "slope_J_s": slope,
        "intercept_J": intercept,
        "h_J_s": h_est,
        "work_function_J": work_function,
        "work_function_joule": work_function,
        "work_function_eV": joule_to_ev(work_function),
        "work_function_ev": joule_to_ev(work_function),
        "cutoff_frequency_Hz": f0,
        "cutoff_wavelength_m": wavelength_from_frequency(f0),
    }


def photoelectric_from_two_wavelength_stopping_data(
    wavelength_1,
    stopping_potential_1,
    wavelength_2,
    stopping_potential_2,
    *,
    c=C,
    e=ELEMENTARY_CHARGE,
):
    """
    Estimate h, phi, and cutoff frequency from two wavelength/V_stop data.
    """
    f1 = frequency_from_wavelength(wavelength_1, c=c)
    f2 = frequency_from_wavelength(wavelength_2, c=c)
    h_est = planck_constant_from_stopping_data(f1, stopping_potential_1, f2, stopping_potential_2, e=e)
    phi = work_function_from_two_stopping_data(f1, stopping_potential_1, f2, stopping_potential_2, e=e)
    f0 = cutoff_frequency_from_two_stopping_data(f1, stopping_potential_1, f2, stopping_potential_2)

    # Canonical keys use explicit units. A few aliases are also returned
    # for convenience and backwards compatibility with tests/examples.
    return {
        "frequency_1_Hz": f1,
        "frequency_2_Hz": f2,
        "h_J_s": h_est,
        "h_estimated": h_est,
        "planck_constant_J_s": h_est,
        "work_function_J": phi,
        "work_function_joule": phi,
        "work_function_eV": joule_to_ev(phi),
        "work_function_ev": joule_to_ev(phi),
        "phi_J": phi,
        "phi_eV": joule_to_ev(phi),
        "cutoff_frequency_Hz": f0,
        "cutoff_frequency": f0,
        "cutoff_wavelength_m": wavelength_from_frequency(f0, c=c),
    }


# ---------------------------------------------------------------------------
# Work function from kinetic-energy ratio
# ---------------------------------------------------------------------------


def work_function_from_two_wavelengths_and_kinetic_ratio(
    wavelength_1,
    wavelength_2,
    kinetic_ratio_2_over_1,
    *,
    h=PLANCK,
    c=C,
):
    """
    Return work function from two wavelengths and a kinetic-energy ratio.

    If
        K1 = h c/lambda1 - phi
        K2 = h c/lambda2 - phi
        K2 = R K1

    then
        phi = (R E1 - E2)/(R - 1)

    where E1 = h c/lambda1 and E2 = h c/lambda2.
    """
    _check_positive(wavelength_1, "wavelength_1")
    _check_positive(wavelength_2, "wavelength_2")
    _check_positive(kinetic_ratio_2_over_1, "kinetic_ratio_2_over_1")

    if _is_numeric_scalar(kinetic_ratio_2_over_1) and float(kinetic_ratio_2_over_1) == 1.0:
        raise ValueError("kinetic_ratio_2_over_1 cannot be 1 for this calculation.")

    E1 = energy_from_wavelength(wavelength_1, h=h, c=c)
    E2 = energy_from_wavelength(wavelength_2, h=h, c=c)
    R = kinetic_ratio_2_over_1

    phi = simplify((R * E1 - E2) / (R - 1))

    if _is_numeric_scalar(phi) and float(phi) < 0.0:
        raise ValueError(
            "The input wavelengths and kinetic ratio imply a negative work function. "
            "Check wavelength order and ratio definition."
        )

    return phi


def work_function_from_two_frequencies_and_kinetic_ratio(
    frequency_1,
    frequency_2,
    kinetic_ratio_2_over_1,
    *,
    h=PLANCK,
):
    """
    Return work function from two frequencies and K2/K1.

    phi = (R E1 - E2)/(R - 1), with E_i = h f_i.
    """
    _check_positive(frequency_1, "frequency_1")
    _check_positive(frequency_2, "frequency_2")
    _check_positive(kinetic_ratio_2_over_1, "kinetic_ratio_2_over_1")

    if _is_numeric_scalar(kinetic_ratio_2_over_1) and float(kinetic_ratio_2_over_1) == 1.0:
        raise ValueError("kinetic_ratio_2_over_1 cannot be 1 for this calculation.")

    E1 = energy_from_frequency(frequency_1, h=h)
    E2 = energy_from_frequency(frequency_2, h=h)
    R = kinetic_ratio_2_over_1
    phi = simplify((R * E1 - E2) / (R - 1))

    if _is_numeric_scalar(phi) and float(phi) < 0.0:
        raise ValueError(
            "The input frequencies and kinetic ratio imply a negative work function. "
            "Check frequency order and ratio definition."
        )

    return phi


# ---------------------------------------------------------------------------
# Electron kinematics helpers
# ---------------------------------------------------------------------------


def electron_speed_from_kinetic_energy_nonrelativistic(
    kinetic_energy,
    *,
    electron_mass=ELECTRON_MASS,
):
    """Return nonrelativistic electron speed v = sqrt(2K/m)."""
    _check_nonnegative(kinetic_energy, "kinetic_energy")
    _check_positive(electron_mass, "electron_mass")

    if is_symbolic(kinetic_energy) and sp is not None:
        return simplify(sp.sqrt(2 * kinetic_energy / electron_mass))
    return float(np.sqrt(2 * kinetic_energy / electron_mass))


def electron_speed_from_stopping_potential_nonrelativistic(
    stopping_potential,
    *,
    e=ELEMENTARY_CHARGE,
    electron_mass=ELECTRON_MASS,
):
    """Return nonrelativistic electron speed from stopping potential."""
    K = kinetic_energy_from_stopping_potential(stopping_potential, e=e)
    return electron_speed_from_kinetic_energy_nonrelativistic(K, electron_mass=electron_mass)


def electron_speed_from_kinetic_energy_relativistic(
    kinetic_energy,
    *,
    electron_mass=ELECTRON_MASS,
    c=C,
):
    """
    Return relativistic electron speed from kinetic energy.

    Uses gamma = 1 + K/(m c^2) and beta = sqrt(1 - 1/gamma^2).
    """
    _check_nonnegative(kinetic_energy, "kinetic_energy")
    _check_positive(electron_mass, "electron_mass")
    _check_positive(c, "c")

    gamma = simplify(1 + kinetic_energy / (electron_mass * c**2))
    beta2 = simplify(1 - 1 / gamma**2)

    if is_symbolic(beta2) and sp is not None:
        beta = sp.sqrt(beta2)
    else:
        beta = np.sqrt(max(float(beta2), 0.0))

    return simplify(c * beta)


def electron_de_broglie_wavelength_nonrelativistic(
    kinetic_energy,
    *,
    h=PLANCK,
    electron_mass=ELECTRON_MASS,
):
    """
    Return nonrelativistic de Broglie wavelength of a photoelectron.

    lambda = h / sqrt(2 m K)
    """
    _check_positive(kinetic_energy, "kinetic_energy")
    _check_positive(h, "h")
    _check_positive(electron_mass, "electron_mass")

    if is_symbolic(kinetic_energy) and sp is not None:
        return simplify(h / sp.sqrt(2 * electron_mass * kinetic_energy))
    return float(h / np.sqrt(2 * electron_mass * kinetic_energy))


# ---------------------------------------------------------------------------
# Convenience summaries
# ---------------------------------------------------------------------------


def photoelectric_summary_from_wavelength(
    wavelength,
    work_function,
    *,
    h=PLANCK,
    c=C,
    e=ELEMENTARY_CHARGE,
    energy_unit: str = "eV",
):
    """
    Return a dictionary with common photoelectric quantities.

    ``work_function`` is expected in joules. Returned energy values are in the
    requested unit, either ``'J'`` or ``'eV'``.
    """
    unit = _check_wavelength_frequency_energy_units(energy_unit)
    frequency = frequency_from_wavelength(wavelength, c=c)
    photon_energy = energy_from_wavelength(wavelength, h=h, c=c)
    kinetic = max_kinetic_energy_from_wavelength(wavelength, work_function, h=h, c=c)
    V_stop = stopping_potential_from_kinetic_energy(kinetic, e=e)

    return {
        "wavelength_m": wavelength,
        "frequency_Hz": frequency,
        f"photon_energy_{unit}": _from_energy_joule(photon_energy, unit),
        f"work_function_{unit}": _from_energy_joule(work_function, unit),
        f"max_kinetic_energy_{unit}": _from_energy_joule(kinetic, unit),
        "stopping_potential_V": V_stop,
        "threshold_frequency_Hz": threshold_frequency(work_function, h=h),
        "threshold_wavelength_m": threshold_wavelength(work_function, h=h, c=c),
    }


def photoelectric_summary_from_frequency(
    frequency,
    work_function,
    *,
    h=PLANCK,
    c=C,
    e=ELEMENTARY_CHARGE,
    energy_unit: str = "eV",
):
    """
    Return a dictionary with common photoelectric quantities from frequency.
    """
    unit = _check_wavelength_frequency_energy_units(energy_unit)
    wavelength = wavelength_from_frequency(frequency, c=c)
    photon_energy = energy_from_frequency(frequency, h=h)
    kinetic = max_kinetic_energy_from_frequency(frequency, work_function, h=h)
    V_stop = stopping_potential_from_kinetic_energy(kinetic, e=e)

    return {
        "frequency_Hz": frequency,
        "wavelength_m": wavelength,
        f"photon_energy_{unit}": _from_energy_joule(photon_energy, unit),
        f"work_function_{unit}": _from_energy_joule(work_function, unit),
        f"max_kinetic_energy_{unit}": _from_energy_joule(kinetic, unit),
        "stopping_potential_V": V_stop,
        "threshold_frequency_Hz": threshold_frequency(work_function, h=h),
        "threshold_wavelength_m": threshold_wavelength(work_function, h=h, c=c),
    }


# Teaching-friendly aliases
max_kinetic_energy = max_kinetic_energy_from_frequency
kinetic_energy_from_frequency = max_kinetic_energy_from_frequency
kinetic_energy_from_wavelength = max_kinetic_energy_from_wavelength
work_function_from_cutoff_frequency = work_function_from_threshold_frequency
work_function_from_cutoff_wavelength = work_function_from_threshold_wavelength


__all__ = [
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "max_kinetic_energy_from_frequency",
    "max_kinetic_energy_from_wavelength",
    "max_kinetic_energy",
    "kinetic_energy_from_frequency",
    "kinetic_energy_from_wavelength",
    "kinetic_energy_from_stopping_potential",
    "stopping_potential_from_kinetic_energy",
    "stopping_potential_from_frequency",
    "stopping_potential_from_wavelength",
    "work_function_from_frequency_and_kinetic_energy",
    "work_function_from_wavelength_and_kinetic_energy",
    "work_function_from_frequency_and_stopping_potential",
    "work_function_from_wavelength_and_stopping_potential",
    "threshold_frequency",
    "cutoff_frequency",
    "threshold_wavelength",
    "cutoff_wavelength",
    "work_function_from_threshold_frequency",
    "work_function_from_threshold_wavelength",
    "work_function_from_cutoff_frequency",
    "work_function_from_cutoff_wavelength",
    "planck_constant_from_stopping_data",
    "work_function_from_two_stopping_data",
    "cutoff_frequency_from_two_stopping_data",
    "photoelectric_line_from_two_points",
    "fit_stopping_potential_data",
    "fit_kinetic_energy_data",
    "photoelectric_from_two_wavelength_stopping_data",
    "work_function_from_two_wavelengths_and_kinetic_ratio",
    "work_function_from_two_frequencies_and_kinetic_ratio",
    "electron_speed_from_kinetic_energy_nonrelativistic",
    "electron_speed_from_stopping_potential_nonrelativistic",
    "electron_speed_from_kinetic_energy_relativistic",
    "electron_de_broglie_wavelength_nonrelativistic",
    "photoelectric_summary_from_wavelength",
    "photoelectric_summary_from_frequency",
    "joule_to_ev",
    "ev_to_joule",
    "nm_to_m",
    "m_to_nm",
]
