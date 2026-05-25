"""Tests for Minkowski products and Lorentz transformations."""
from __future__ import annotations

import numpy as np
import sympy as sp

from relativity.math.minkowski import minkowski_dot, spacetime_interval, classify_interval
from relativity.math.lorentz import boost_matrix, lorentz_transform_fourvector, velocity_addition_1d


def test_minkowski_dot_numeric_signature_plus_minus_minus_minus():
    a = [2, 1, 0, 0]
    assert minkowski_dot(a, a) == 3


def test_spacetime_interval_and_classification():
    assert spacetime_interval([2, 1, 0, 0]) == 3
    assert classify_interval([2, 1, 0, 0]) == "timelike"
    assert classify_interval([1, 2, 0, 0]) == "spacelike"
    assert classify_interval([1, 1, 0, 0]) == "lightlike"


def test_lorentz_interval_invariant_numeric():
    c = 1.0
    X = np.array([2.0, 0.6, 0.1, 0.0])
    v = np.array([0.5, 0.0, 0.0])
    Xp = lorentz_transform_fourvector(X, v, c)
    assert np.isclose(minkowski_dot(X, X), minkowski_dot(Xp, Xp))


def test_lorentz_boost_matrix_symbolic_shape_and_entries():
    v, c = sp.symbols("v c", positive=True)
    Lambda = boost_matrix([v, 0, 0], c)
    assert Lambda.shape == (4, 4)
    expected_gamma = 1 / sp.sqrt(1 - v**2/c**2)
    assert sp.simplify(Lambda[0, 0] - expected_gamma) == 0
    assert sp.simplify(Lambda[0, 1] + expected_gamma * v/c) == 0


def test_velocity_addition_1d_low_speed_limit():
    assert np.isclose(velocity_addition_1d(0.1, 0.2, c=1.0), (0.1 - 0.2) / (1 - 0.02))
