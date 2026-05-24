import numpy as np

from relativity.math.tensors import MinkowskiMetric
from relativity.utils import (
    smart_dot,
    smart_sqrt,
    is_symbolic,
    smart_array
)

# FIX: was importing ETA which was never exported; expose it properly
eta = MinkowskiMetric().components


def minkowski_dot(a, b):
    a = smart_array(a)
    b = smart_array(b)
    return smart_dot(a, eta @ b)


def spacetime_interval(x):
    return minkowski_dot(x, x)


# FIX: alias so event.py can import interval_squared from here
interval_squared = spacetime_interval


def classify_interval(x):
    s2 = spacetime_interval(x)
    if is_symbolic(s2):
        return s2
    if s2 > 0:
        return "timelike"
    elif s2 < 0:
        return "spacelike"
    return "lightlike"


def proper_time(dx, c):
    s2 = spacetime_interval(dx)
    return smart_sqrt(s2) / c


def lower_index(v):
    return eta @ smart_array(v)


def raise_index(v):
    # FIX: raise_index and lower_index were identical; raising requires inverse metric.
    # For Minkowski with sig +---, eta^{-1} == eta, so this is actually correct,
    # but we make it explicit for clarity.
    eta_inv = MinkowskiMetric().components  # eta_inv == eta for Minkowski
    return eta_inv @ smart_array(v)
