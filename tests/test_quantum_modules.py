"""
Tests for relativity.quantum modules.

Place this file inside your project's tests/ directory and run:

    python -m pytest tests/test_quantum_modules.py -q

These tests assume the quantum package contains:
    photons.py, blackbody.py, photoelectric.py, compton.py,
    de_broglie.py, uncertainty.py, probability.py
"""

import math

import numpy as np
import pytest


def test_quantum_modules_import():
    from relativity.quantum import (  # noqa: F401
        photons,
        blackbody,
        photoelectric,
        compton,
        de_broglie,
        uncertainty,
        probability,
    )


# -----------------------------------------------------------------------------
# photons.py
# -----------------------------------------------------------------------------


def test_photon_energy_frequency_wavelength_roundtrip():
    from relativity.quantum import photons

    wavelength = 500e-9
    frequency = photons.frequency_from_wavelength(wavelength)
    energy = photons.energy_from_wavelength(wavelength)

    assert frequency == pytest.approx(photons.C / wavelength)
    assert energy == pytest.approx(photons.PLANCK * photons.C / wavelength)
    assert photons.wavelength_from_frequency(frequency) == pytest.approx(wavelength)
    assert photons.wavelength_from_energy(energy) == pytest.approx(wavelength)


def test_photon_joule_ev_roundtrip_and_count():
    from relativity.quantum import photons

    energy_j = 1.0
    energy_ev = photons.joule_to_ev(energy_j)
    assert photons.ev_to_joule(energy_ev) == pytest.approx(energy_j)

    wavelength = 500e-9
    total_energy = 98.1
    n = photons.photon_count_from_total_energy(total_energy, wavelength=wavelength)
    expected = total_energy / photons.energy_from_wavelength(wavelength)
    assert n == pytest.approx(expected)


def test_photon_rejects_zero_direction_for_momentum_vector():
    from relativity.quantum import photons

    with pytest.raises(ValueError):
        photons.momentum_vector_from_wavelength(500e-9, direction=[0, 0, 0])


# -----------------------------------------------------------------------------
# blackbody.py
# -----------------------------------------------------------------------------


def test_blackbody_stefan_boltzmann_and_sphere_luminosity():
    from relativity.quantum import blackbody

    temperature = 5700.0
    diameter = 1.4e9

    flux = blackbody.stefan_boltzmann_flux(temperature)
    assert flux == pytest.approx(blackbody.SIGMA_SB * temperature**4)

    power = blackbody.luminosity_sphere(temperature, diameter=diameter)
    expected_area = math.pi * diameter**2
    assert power == pytest.approx(expected_area * blackbody.SIGMA_SB * temperature**4)

    dm_dt = blackbody.mass_loss_rate_from_power(power)
    assert dm_dt == pytest.approx(power / blackbody.C**2)


def test_blackbody_wien_law_roundtrip():
    from relativity.quantum import blackbody

    temperature = 1.0e7
    wavelength_peak = blackbody.wien_peak_wavelength(temperature)

    assert wavelength_peak == pytest.approx(blackbody.WIEN_DISPLACEMENT / temperature)
    assert blackbody.temperature_from_peak_wavelength(wavelength_peak) == pytest.approx(temperature)


# -----------------------------------------------------------------------------
# photoelectric.py
# -----------------------------------------------------------------------------


def test_photoelectric_work_function_from_two_wavelengths_ratio():
    from relativity.quantum import photoelectric

    lambda_red = 670e-9
    lambda_green = 520e-9

    phi = photoelectric.work_function_from_two_wavelengths_and_kinetic_ratio(
        lambda_red,
        lambda_green,
        kinetic_ratio_2_over_1=1.5,
    )

    k_red = photoelectric.max_kinetic_energy_from_wavelength(lambda_red, phi)
    k_green = photoelectric.max_kinetic_energy_from_wavelength(lambda_green, phi)

    assert phi > 0
    assert k_red > 0
    assert k_green / k_red == pytest.approx(1.5)


def test_photoelectric_two_stopping_data_estimates_reasonable_h():
    from relativity.quantum import photoelectric

    result = photoelectric.photoelectric_from_two_wavelength_stopping_data(
        wavelength_1=254e-9,
        stopping_potential_1=3.00,
        wavelength_2=436e-9,
        stopping_potential_2=0.900,
    )

    assert result["h_estimated"] == pytest.approx(6.64e-34, rel=0.03)
    assert result["cutoff_frequency"] > 0
    assert result["work_function_joule"] > 0


# -----------------------------------------------------------------------------
# compton.py
# -----------------------------------------------------------------------------


def test_compton_wavelength_and_max_shift():
    from relativity.quantum import compton

    lambda_c = compton.electron_compton_wavelength()

    assert lambda_c == pytest.approx(2.42631023867e-12, rel=1e-5)
    assert compton.max_compton_shift() == pytest.approx(2.0 * lambda_c)


def test_compton_backscatter_and_blue_light_fractional_shift():
    from relativity.quantum import compton

    wavelength = 480e-9
    frac = compton.max_fractional_compton_shift(wavelength)
    expected = 2.0 * compton.electron_compton_wavelength() / wavelength

    assert frac == pytest.approx(expected)
    assert frac < 2e-5

    shifted = compton.scattered_wavelength(wavelength, theta=180, degrees=True)
    assert shifted == pytest.approx(wavelength + 2.0 * compton.electron_compton_wavelength())


# -----------------------------------------------------------------------------
# de_broglie.py
# -----------------------------------------------------------------------------


def test_de_broglie_electron_wavelength_from_speed():
    from relativity.quantum import de_broglie

    speed = 3.0e6
    wavelength = de_broglie.electron_wavelength_from_speed(speed)

    gamma = 1.0 / math.sqrt(1.0 - (speed / de_broglie.C) ** 2)
    expected = de_broglie.PLANCK / (gamma * de_broglie.ELECTRON_MASS * speed)

    assert wavelength == pytest.approx(expected)


def test_de_broglie_nonrelativistic_close_at_low_speed():
    from relativity.quantum import de_broglie

    speed = 3.0e6
    wavelength_rel = de_broglie.wavelength_from_speed(
        de_broglie.ELECTRON_MASS,
        speed,
        relativistic=True,
    )
    wavelength_nr = de_broglie.wavelength_from_speed(
        de_broglie.ELECTRON_MASS,
        speed,
        relativistic=False,
    )

    assert wavelength_rel == pytest.approx(wavelength_nr, rel=1e-3)


# -----------------------------------------------------------------------------
# uncertainty.py
# -----------------------------------------------------------------------------


def test_uncertainty_energy_time_for_gamma_lifetime():
    from relativity.quantum import uncertainty

    lifetime = 1.0e-12
    dE = uncertainty.energy_uncertainty_from_lifetime(lifetime)

    assert dE == pytest.approx(uncertainty.HBAR / (2.0 * lifetime))
    assert uncertainty.joule_to_ev(dE) == pytest.approx(3.291e-4, rel=5e-3)


def test_uncertainty_position_momentum_product_bound():
    from relativity.quantum import uncertainty

    dx = 1.0e-10
    dp = uncertainty.momentum_uncertainty_from_position(dx)

    assert dp == pytest.approx(uncertainty.HBAR / (2.0 * dx))
    assert uncertainty.satisfies_uncertainty(dx, dp)


# -----------------------------------------------------------------------------
# probability.py
# -----------------------------------------------------------------------------


def test_probability_tarea3_exercise10_summary():
    from relativity.quantum import probability

    result = probability.tarea3_exercise10_summary()

    assert float(result["continuous_mean"]) == pytest.approx(2.5)
    assert float(result["discrete"][1]["mean"]) == pytest.approx(15.0 / 7.0)
    assert float(result["discrete"][5]["mean"]) == pytest.approx(1.0)


def test_probability_continuous_and_discrete_average_helpers():
    from relativity.quantum import probability

    f = lambda x: (10.0 - x) ** 2 / 10.0

    continuous = probability.numeric_average(f, 0.0, 10.0, num=20001)
    discrete = probability.discrete_average_from_function([0, 1, 2, 3], f)

    assert continuous == pytest.approx(2.5, rel=1e-4)
    assert discrete >= 0


# -----------------------------------------------------------------------------
# Symbolic smoke tests
# -----------------------------------------------------------------------------


def test_quantum_symbolic_smoke_tests():
    sp = pytest.importorskip("sympy")

    from relativity.quantum import photons, compton, probability

    lam, h, c = sp.symbols("lambda h c", positive=True)
    assert photons.energy_from_wavelength(lam, h=h, c=c) == h * c / lam

    theta, m = sp.symbols("theta m", positive=True)
    shift = compton.compton_shift(theta, particle_mass=m, h=h, c=c)
    assert sp.simplify(shift - h / (m * c) * (1 - sp.cos(theta))) == 0

    x = sp.symbols("x", real=True)
    mean = probability.continuous_average((10 - x) ** 2 / 10, x, 0, 10)
    assert sp.simplify(mean - sp.Rational(5, 2)) == 0
