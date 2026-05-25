"""Minkowski-space utilities with signature (+---)."""
from __future__ import annotations
from relativity.math.tensors import MinkowskiMetric
from relativity.utils import smart_array, smart_dot, smart_sqrt, is_symbolic, simplify

ETA = MinkowskiMetric().components
eta = ETA


def minkowski_dot(a, b):
    a, b = smart_array(a), smart_array(b)
    return simplify(smart_dot(a, ETA @ b))


def interval_squared(x):
    return minkowski_dot(x, x)


def spacetime_interval(x):
    return interval_squared(x)


def classify_interval(x):
    s2 = interval_squared(x)
    if is_symbolic(s2):
        return simplify(s2)
    if s2 > 0: return "timelike"
    if s2 < 0: return "spacelike"
    return "lightlike"


def proper_time(dx, c):
    s2 = interval_squared(dx)
    if not is_symbolic(s2) and s2 < 0:
        raise ValueError("Intervalo tipo espacio: no existe tiempo propio real.")
    return smart_sqrt(s2) / c


def proper_length(dx):
    s2 = interval_squared(dx)
    if not is_symbolic(s2) and s2 > 0:
        raise ValueError("Intervalo tipo tiempo: no existe longitud propia espacial.")
    return smart_sqrt(-s2)


def lower_index(v): return ETA @ smart_array(v)
def raise_index(v): return ETA @ smart_array(v)
