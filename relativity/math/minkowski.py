"""Minkowski-space helpers using signature (+---)."""

from __future__ import annotations

import numpy as np

from relativity.math.tensors import MinkowskiMetric
from relativity.utils import smart_array, smart_dot, smart_sqrt, smart_matmul, is_symbolic, simplify

eta = MinkowskiMetric().components


def minkowski_dot(a, b):
    a = smart_array(a)
    b = smart_array(b)
    return simplify(smart_dot(a, smart_matmul(eta, b)))


def spacetime_interval(x):
    return minkowski_dot(x, x)


interval_squared = spacetime_interval


def classify_interval(x):
    s2 = spacetime_interval(x)
    if is_symbolic(s2):
        return s2
    if s2 > 0:
        return "timelike"
    if s2 < 0:
        return "spacelike"
    return "lightlike"


def proper_time(dx, c):
    s2 = spacetime_interval(dx)
    if not is_symbolic(s2) and s2 < 0:
        raise ValueError("Intervalo space-like: no hay tiempo propio real.")
    return simplify(smart_sqrt(s2) / c)


def lower_index(v):
    return smart_matmul(eta, smart_array(v))


def raise_index(v):
    # For signature (+---), eta^{-1} = eta.
    return smart_matmul(eta, smart_array(v))
