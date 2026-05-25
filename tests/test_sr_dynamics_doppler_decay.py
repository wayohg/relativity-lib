"""Tests for dynamics, Doppler, and decay helpers."""
from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from relativity.sr import dynamics as dyn
from relativity.sr import doppler
from relativity.sr import decay


def test_energy_momentum_relation_numeric():
    m = 2.0
    v = [0.6, 0, 0]
    c = 1.0
    E = dyn.total_energy(m, v, c=c)
    p = dyn.momentum(m, v, c=c)
    assert np.isclose(E**2, np.dot(p, p)*c**2 + m**2*c**4)


def test_invariant_mass_from_energy_momentum():
    m = 2.0
    v = [0.6, 0, 0]
    c = 1.0
    E = dyn.total_energy(m, v, c=c)
    p = dyn.momentum(m, v, c=c)
    assert np.isclose(dyn.invariant_mass_from_energy_momentum(E, p, c=c), m)


def test_dynamics_symbolic_total_energy():
    m, v, c = sp.symbols("m v c", positive=True)
    E = dyn.total_energy(m, [v, 0, 0], c=c)
    expected = m*c**2 / sp.sqrt(1 - v**2/c**2)
    assert sp.simplify(E - expected) == 0


def test_longitudinal_doppler_factor_receding():
    beta = 0.5
    expected = np.sqrt((1 - beta) / (1 + beta))
    assert np.isclose(doppler.longitudinal_doppler_factor(beta), expected)


def test_redshift_inverts_beta_for_longitudinal_case():
    beta = 0.3
    D = doppler.longitudinal_doppler_factor(beta, approaching=False)
    z = doppler.redshift_from_doppler(D)
    assert np.isclose(doppler.beta_from_longitudinal_redshift(z), beta)


def test_lab_lifetime_and_decay_length():
    tau0 = 2.0
    v = 0.6
    c = 1.0
    gamma = 1.25
    assert np.isclose(decay.lab_lifetime(tau0, v, c=c), gamma*tau0)
    assert np.isclose(decay.decay_length(tau0, v, c=c), v*gamma*tau0)


def test_survival_probability_time_basic():
    assert np.isclose(decay.survival_probability_time(1.0, lab_lifetime_value=1.0), np.exp(-1.0))


def test_two_body_decay_equal_mass_products():
    p = decay.two_body_decay_momentum(4.0, 1.0, 1.0, c=1.0)
    assert p > 0
    E1, E2 = decay.two_body_decay_energies(4.0, 1.0, 1.0, c=1.0)
    assert np.isclose(E1 + E2, 4.0)


def test_two_body_decay_disallowed_raises():
    with pytest.raises(ValueError):
        decay.two_body_decay_momentum(1.0, 1.0, 1.0, c=1.0)
