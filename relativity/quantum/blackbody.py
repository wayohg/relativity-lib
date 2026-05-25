"""
relativity.quantum.blackbody
============================

Herramientas para radiación de cuerpo negro y radiación térmica.

Este módulo está pensado para complementar la capa ``quantum`` de la librería.
Usa unidades SI por defecto:

- Temperatura: kelvin (K)
- Longitud de onda: metro (m)
- Frecuencia: hertz (Hz)
- Potencia: watt (W)
- Energía: joule (J)
- Masa: kilogramo (kg)

Incluye funciones útiles para ejercicios introductorios de física moderna:

- Ley de Stefan-Boltzmann
- Ley de desplazamiento de Wien
- Radiancia espectral de Planck en longitud de onda y frecuencia
- Luminosidad de una esfera emisora
- Equivalencia masa-energía para pérdida de masa por radiación

Ejemplo
-------
>>> from relativity.quantum.blackbody import luminosity_sphere, mass_loss_rate_from_power
>>> T_sun = 5700
>>> diameter_sun = 1.4e9
>>> P = luminosity_sphere(T_sun, diameter=diameter_sun)
>>> dm_dt = mass_loss_rate_from_power(P)
>>> P, dm_dt

Notas
-----
Para graficar espectros, genera un arreglo de longitudes de onda con NumPy y
pásalo a ``planck_radiance_wavelength``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover - sympy should normally exist in this project
    sp = None


# -----------------------------------------------------------------------------
# Constants with safe fallbacks
# -----------------------------------------------------------------------------

try:
    from relativity import constants as _constants
except Exception:  # pragma: no cover
    _constants = None


def _const(*names: str, default: float) -> float:
    """Read a constant from relativity.constants with several possible names."""
    if _constants is not None:
        for name in names:
            if hasattr(_constants, name):
                return getattr(_constants, name)
    return default


C = _const("C", "c", "SPEED_OF_LIGHT", default=299_792_458.0)
H = _const("PLANCK", "H", "h", "PLANCK_CONSTANT", default=6.62607015e-34)
K_B = _const("BOLTZMANN", "K_B", "KB", "k_B", default=1.380649e-23)
SIGMA_SB = _const(
    "STEFAN_BOLTZMANN",
    "SIGMA_SB",
    "STEFAN_BOLTZMANN_CONSTANT",
    "SIGMA",
    default=5.670374419e-8,
)
WIEN_DISPLACEMENT = _const(
    "WIEN_DISPLACEMENT",
    "WIEN",
    "WIEN_CONSTANT",
    default=2.897771955e-3,
)

# Maximum of Planck spectrum per unit frequency: h nu / kT = 2.821439...
WIEN_FREQUENCY_FACTOR = 2.8214393721220787

Number = Union[int, float, np.ndarray]
ScalarLike = Union[int, float]


# -----------------------------------------------------------------------------
# Optional integration with relativity.utils
# -----------------------------------------------------------------------------

try:
    from relativity.utils import is_symbolic, simplify
except Exception:

    def is_symbolic(x) -> bool:
        if sp is None:
            return False
        try:
            if isinstance(x, np.ndarray):
                return any(is_symbolic(item) for item in x.flat)
            return bool(getattr(x, "free_symbols", set())) or isinstance(x, sp.Basic)
        except Exception:
            return False

    def simplify(x):
        if sp is not None and is_symbolic(x):
            return sp.simplify(x)
        return x


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _is_array_like(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _validate_positive(value, name: str, allow_zero: bool = False) -> None:
    """Validate positive numeric input while allowing symbolic expressions."""
    if is_symbolic(value):
        return

    if _is_array_like(value):
        arr = np.asarray(value, dtype=float)
        if allow_zero:
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative.")
        else:
            if np.any(arr <= 0):
                raise ValueError(f"{name} must be positive.")
        return

    value_float = float(value)
    if allow_zero:
        if value_float < 0:
            raise ValueError(f"{name} must be non-negative.")
    else:
        if value_float <= 0:
            raise ValueError(f"{name} must be positive.")


def _exp(x):
    """Exponential that works with symbolic and numeric values."""
    if is_symbolic(x):
        if sp is None:
            raise RuntimeError("SymPy is required for symbolic expressions.")
        return sp.exp(x)
    return np.exp(x)


def _expm1(x):
    """exp(x)-1 with better numerical stability."""
    if is_symbolic(x):
        if sp is None:
            raise RuntimeError("SymPy is required for symbolic expressions.")
        return sp.exp(x) - 1
    return np.expm1(x)


# -----------------------------------------------------------------------------
# Stefan-Boltzmann law
# -----------------------------------------------------------------------------


def stefan_boltzmann_flux(
    temperature,
    emissivity=1.0,
    sigma: float = SIGMA_SB,
):
    """
    Radiant exitance/flux of a blackbody surface.

    Formula
    -------
    M = epsilon * sigma * T^4

    Parameters
    ----------
    temperature : float or symbolic
        Absolute temperature in K.
    emissivity : float or symbolic, optional
        Dimensionless emissivity. For an ideal blackbody, emissivity = 1.
    sigma : float, optional
        Stefan-Boltzmann constant.

    Returns
    -------
    float or symbolic
        Flux in W/m^2.
    """
    _validate_positive(temperature, "temperature")
    _validate_positive(emissivity, "emissivity")
    return simplify(emissivity * sigma * temperature**4)


def luminosity_from_area(
    temperature,
    area,
    emissivity=1.0,
    sigma: float = SIGMA_SB,
):
    """
    Total emitted power from area A at temperature T.

    P = epsilon * sigma * A * T^4
    """
    _validate_positive(area, "area")
    return simplify(area * stefan_boltzmann_flux(temperature, emissivity, sigma))


def sphere_area(
    radius: Optional[float] = None,
    diameter: Optional[float] = None,
):
    """
    Surface area of a sphere.

    Provide either ``radius`` or ``diameter``.
    """
    if radius is None and diameter is None:
        raise ValueError("Provide either radius or diameter.")
    if radius is not None and diameter is not None:
        raise ValueError("Provide radius or diameter, not both.")

    if radius is None:
        _validate_positive(diameter, "diameter")
        radius = diameter / 2
    else:
        _validate_positive(radius, "radius")

    return simplify(4 * math.pi * radius**2)


def luminosity_sphere(
    temperature,
    radius: Optional[float] = None,
    diameter: Optional[float] = None,
    emissivity=1.0,
    sigma: float = SIGMA_SB,
):
    """
    Luminosity of a spherical blackbody emitter.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    radius, diameter : float, optional
        Sphere size in m. Provide exactly one.
    emissivity : float, optional
        Emissivity. Use 1 for ideal blackbody.
    """
    area = sphere_area(radius=radius, diameter=diameter)
    return luminosity_from_area(temperature, area, emissivity, sigma)


def radiated_energy(power, time):
    """Energy emitted over time: E = P t."""
    _validate_positive(power, "power", allow_zero=True)
    _validate_positive(time, "time", allow_zero=True)
    return simplify(power * time)


def mass_equivalent(energy, c: float = C):
    """Mass equivalent of energy: m = E / c^2."""
    _validate_positive(energy, "energy", allow_zero=True)
    _validate_positive(c, "c")
    return simplify(energy / c**2)


def energy_equivalent(mass, c: float = C):
    """Energy equivalent of mass: E = m c^2."""
    _validate_positive(mass, "mass", allow_zero=True)
    _validate_positive(c, "c")
    return simplify(mass * c**2)


def mass_loss_rate_from_power(power, c: float = C):
    """
    Rest-mass loss rate due to radiation.

    dm/dt = P / c^2
    """
    _validate_positive(power, "power", allow_zero=True)
    return mass_equivalent(power, c=c)


def mass_loss_rate_blackbody(
    temperature,
    area: Optional[float] = None,
    radius: Optional[float] = None,
    diameter: Optional[float] = None,
    emissivity=1.0,
    sigma: float = SIGMA_SB,
    c: float = C,
):
    """
    Mass loss rate for a radiating blackbody.

    Provide either an explicit ``area`` or a sphere ``radius``/``diameter``.
    """
    if area is None:
        power = luminosity_sphere(
            temperature,
            radius=radius,
            diameter=diameter,
            emissivity=emissivity,
            sigma=sigma,
        )
    else:
        if radius is not None or diameter is not None:
            raise ValueError("Use either area or sphere radius/diameter, not both.")
        power = luminosity_from_area(temperature, area, emissivity, sigma)

    return mass_loss_rate_from_power(power, c=c)


def fractional_mass_loss(
    mass,
    emitted_energy=None,
    power=None,
    time=None,
    c: float = C,
):
    """
    Fraction of rest mass lost to radiation.

    Either provide ``emitted_energy`` directly, or provide ``power`` and ``time``.
    """
    _validate_positive(mass, "mass")

    if emitted_energy is None:
        if power is None or time is None:
            raise ValueError("Provide emitted_energy or both power and time.")
        emitted_energy = radiated_energy(power, time)

    lost_mass = mass_equivalent(emitted_energy, c=c)
    return simplify(lost_mass / mass)


def fractional_mass_loss_per_year(
    mass,
    power,
    seconds_per_year: float = 365.25 * 24 * 3600,
    c: float = C,
):
    """Fraction of mass lost in one year for a constant radiated power."""
    return fractional_mass_loss(mass, power=power, time=seconds_per_year, c=c)


# -----------------------------------------------------------------------------
# Wien displacement law
# -----------------------------------------------------------------------------


def wien_peak_wavelength(
    temperature,
    b: float = WIEN_DISPLACEMENT,
):
    """
    Wavelength of maximum spectral radiance per unit wavelength.

    lambda_max = b / T
    """
    _validate_positive(temperature, "temperature")
    return simplify(b / temperature)


def temperature_from_peak_wavelength(
    wavelength,
    b: float = WIEN_DISPLACEMENT,
):
    """
    Temperature from wavelength peak using Wien's displacement law.

    T = b / lambda_max
    """
    _validate_positive(wavelength, "wavelength")
    return simplify(b / wavelength)


def wien_peak_frequency(
    temperature,
    h: float = H,
    k_B: float = K_B,
):
    """
    Frequency of maximum spectral radiance per unit frequency.

    Important
    ---------
    This is not simply c / lambda_max, because the maxima of B_lambda and
    B_nu occur at different points.

    nu_max = x * k_B T / h, where x ≈ 2.821439.
    """
    _validate_positive(temperature, "temperature")
    return simplify(WIEN_FREQUENCY_FACTOR * k_B * temperature / h)


def temperature_from_peak_frequency(
    frequency,
    h: float = H,
    k_B: float = K_B,
):
    """Temperature from the peak frequency of B_nu."""
    _validate_positive(frequency, "frequency")
    return simplify(frequency * h / (WIEN_FREQUENCY_FACTOR * k_B))


# -----------------------------------------------------------------------------
# Planck radiation law
# -----------------------------------------------------------------------------


def planck_radiance_wavelength(
    wavelength,
    temperature,
    h: float = H,
    c: float = C,
    k_B: float = K_B,
):
    r"""
    Spectral radiance of a blackbody per unit wavelength.

    Formula
    -------
    B_λ(λ, T) = (2 h c² / λ⁵) / (exp(h c / (λ k_B T)) - 1)

    Units
    -----
    W sr^-1 m^-3
    """
    _validate_positive(wavelength, "wavelength")
    _validate_positive(temperature, "temperature")

    x = h * c / (wavelength * k_B * temperature)
    return simplify((2 * h * c**2 / wavelength**5) / _expm1(x))


def planck_radiance_frequency(
    frequency,
    temperature,
    h: float = H,
    c: float = C,
    k_B: float = K_B,
):
    r"""
    Spectral radiance of a blackbody per unit frequency.

    Formula
    -------
    B_ν(ν, T) = (2 h ν³ / c²) / (exp(hν / (k_B T)) - 1)

    Units
    -----
    W sr^-1 m^-2 Hz^-1
    """
    _validate_positive(frequency, "frequency")
    _validate_positive(temperature, "temperature")

    x = h * frequency / (k_B * temperature)
    return simplify((2 * h * frequency**3 / c**2) / _expm1(x))


def spectral_exitance_wavelength(
    wavelength,
    temperature,
    h: float = H,
    c: float = C,
    k_B: float = K_B,
):
    """
    Spectral exitance per unit wavelength for a Lambertian blackbody.

    M_lambda = pi * B_lambda
    """
    return simplify(math.pi * planck_radiance_wavelength(wavelength, temperature, h, c, k_B))


def spectral_exitance_frequency(
    frequency,
    temperature,
    h: float = H,
    c: float = C,
    k_B: float = K_B,
):
    """
    Spectral exitance per unit frequency for a Lambertian blackbody.

    M_nu = pi * B_nu
    """
    return simplify(math.pi * planck_radiance_frequency(frequency, temperature, h, c, k_B))


def rayleigh_jeans_radiance_wavelength(
    wavelength,
    temperature,
    c: float = C,
    k_B: float = K_B,
):
    """
    Rayleigh-Jeans approximation for long wavelengths.

    B_lambda ≈ 2 c k_B T / lambda^4
    """
    _validate_positive(wavelength, "wavelength")
    _validate_positive(temperature, "temperature")
    return simplify(2 * c * k_B * temperature / wavelength**4)


def wien_approx_radiance_wavelength(
    wavelength,
    temperature,
    h: float = H,
    c: float = C,
    k_B: float = K_B,
):
    """
    Wien approximation for short wavelengths.

    B_lambda ≈ (2 h c^2 / lambda^5) exp[-h c / (lambda k_B T)]
    """
    _validate_positive(wavelength, "wavelength")
    _validate_positive(temperature, "temperature")
    x = h * c / (wavelength * k_B * temperature)
    return simplify((2 * h * c**2 / wavelength**5) / _exp(x))


# -----------------------------------------------------------------------------
# Convenience summaries
# -----------------------------------------------------------------------------


def blackbody_summary(
    temperature,
    radius: Optional[float] = None,
    diameter: Optional[float] = None,
    emissivity=1.0,
    mass: Optional[float] = None,
    c: float = C,
):
    """
    Return a dictionary with common blackbody quantities.

    If ``radius`` or ``diameter`` is provided, the result includes surface area,
    luminosity and mass loss rate. If ``mass`` is also provided, it includes the
    fractional mass loss per year.
    """
    result = {
        "temperature_K": temperature,
        "peak_wavelength_m": wien_peak_wavelength(temperature),
        "peak_frequency_Hz": wien_peak_frequency(temperature),
        "flux_W_m2": stefan_boltzmann_flux(temperature, emissivity=emissivity),
    }

    if radius is not None or diameter is not None:
        area = sphere_area(radius=radius, diameter=diameter)
        power = luminosity_sphere(
            temperature,
            radius=radius,
            diameter=diameter,
            emissivity=emissivity,
        )
        result.update(
            {
                "area_m2": area,
                "luminosity_W": power,
                "mass_loss_rate_kg_s": mass_loss_rate_from_power(power, c=c),
            }
        )
        if mass is not None:
            result["fractional_mass_loss_per_year"] = fractional_mass_loss_per_year(
                mass,
                power,
                c=c,
            )

    return result


def sample_blackbody_spectrum_wavelength(
    temperature,
    wavelength_min: Optional[float] = None,
    wavelength_max: Optional[float] = None,
    num: int = 1000,
    around_peak_factor: Tuple[float, float] = (0.1, 10.0),
):
    """
    Generate wavelength and Planck-radiance arrays for plotting.

    If limits are not provided, the range is chosen around the Wien peak.
    """
    if num < 2:
        raise ValueError("num must be at least 2.")

    peak = float(wien_peak_wavelength(temperature))

    if wavelength_min is None:
        wavelength_min = around_peak_factor[0] * peak
    if wavelength_max is None:
        wavelength_max = around_peak_factor[1] * peak

    _validate_positive(wavelength_min, "wavelength_min")
    _validate_positive(wavelength_max, "wavelength_max")
    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max.")

    wavelengths = np.linspace(wavelength_min, wavelength_max, num)
    radiance = planck_radiance_wavelength(wavelengths, temperature)
    return wavelengths, radiance


__all__ = [
    "C",
    "H",
    "K_B",
    "SIGMA_SB",
    "WIEN_DISPLACEMENT",
    "WIEN_FREQUENCY_FACTOR",
    "stefan_boltzmann_flux",
    "luminosity_from_area",
    "sphere_area",
    "luminosity_sphere",
    "radiated_energy",
    "mass_equivalent",
    "energy_equivalent",
    "mass_loss_rate_from_power",
    "mass_loss_rate_blackbody",
    "fractional_mass_loss",
    "fractional_mass_loss_per_year",
    "wien_peak_wavelength",
    "temperature_from_peak_wavelength",
    "wien_peak_frequency",
    "temperature_from_peak_frequency",
    "planck_radiance_wavelength",
    "planck_radiance_frequency",
    "spectral_exitance_wavelength",
    "spectral_exitance_frequency",
    "rayleigh_jeans_radiance_wavelength",
    "wien_approx_radiance_wavelength",
    "blackbody_summary",
    "sample_blackbody_spectrum_wavelength",
]
