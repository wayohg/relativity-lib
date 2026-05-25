"""Tests for relativity.sr.kinematics."""
from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from relativity.sr import kinematics as kin


def test_gamma_numeric_beta():
    assert np.isclose(kin.gamma(beta_value=0.6), 1.25)


def test_gamma_numeric_vector_c_equals_one():
    assert np.isclose(kin.gamma([0.6, 0.0, 0.0], c=1.0), 1.25)


def test_gamma_rejects_superluminal_numeric():
    with pytest.raises(ValueError):
        kin.gamma([1.0, 0.0, 0.0], c=1.0)


def test_gamma_symbolic_beta():
    b = sp.Symbol("b")
    g = kin.gamma(beta_value=b)
    assert sp.simplify(g - 1/sp.sqrt(1 - b**2)) == 0


def test_time_dilation_and_length_contraction():
    v = [0.6, 0, 0]
    assert np.isclose(kin.dilated_time(2.0, v, c=1.0), 2.5)
    assert np.isclose(kin.contracted_length(10.0, v, c=1.0), 8.0)


def test_lorentz_transform_event_1d_interval_invariance():
    t, x, v, c = 2.0, 0.7, 0.5, 1.0
    tp, xp = kin.lorentz_transform_event_1d(t, x, v, c)
    assert np.isclose(c**2*t**2 - x**2, c**2*tp**2 - xp**2)


def test_velocity_addition_1d_subluminal():
    u_prime = kin.velocity_addition_1d(0.8, 0.5, c=1.0)
    assert abs(u_prime) < 1.0


def test_separation_classification():
    assert kin.classify_separation(0, [0, 0, 0], 2, [1, 0, 0], c=1) == "timelike"
    assert kin.classify_separation(0, [0, 0, 0], 1, [2, 0, 0], c=1) == "spacelike"
    assert kin.classify_separation(0, [0, 0, 0], 1, [1, 0, 0], c=1) == "lightlike"
