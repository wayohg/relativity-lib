"""Lorentz transformations and related helpers."""
from __future__ import annotations
import numpy as np
from relativity.utils import smart_array, smart_sqrt, gamma_factor, smart_dot, is_symbolic


def gamma(v, c=1.0):
    return gamma_factor(smart_array(v), c)


def boost_matrix(v, c=1.0):
    """Matrix for coordinates from S to S' where S' moves with velocity v wrt S."""
    v = smart_array(v)
    v2 = smart_dot(v, v)
    if not is_symbolic(v2) and abs(v2) == 0:
        return np.eye(4)
    g = gamma(v, c)
    beta = v / c
    beta2 = v2 / c**2
    dtype = object if is_symbolic(v) or is_symbolic(g) else float
    L = np.eye(4, dtype=dtype)
    L[0, 0] = g
    L[0, 1:] = -g * beta
    L[1:, 0] = -g * beta
    L[1:, 1:] += ((g - 1) / beta2) * np.outer(beta, beta)
    return L


def inverse_boost_matrix(v, c=1.0):
    return boost_matrix(-smart_array(v), c)


def transform_fourvector(x, v, c=1.0):
    return boost_matrix(v, c) @ smart_array(x)


def relativistic_velocity_addition(u, v, c=1.0):
    """Velocity of object u transformed by adding frame velocity v, 3D form."""
    u, v = smart_array(u), smart_array(v)
    v2 = smart_dot(v, v)
    if not is_symbolic(v2) and v2 == 0:
        return u
    dotuv = smart_dot(u, v)
    u_parallel = (dotuv / v2) * v
    u_perp = u - u_parallel
    gv = gamma(v, c)
    denom = 1 + dotuv / c**2
    return (u_parallel + v) / denom + u_perp / (gv * denom)


def time_dilation(proper_time, v, c=1.0):
    return gamma(v, c) * proper_time


def length_contraction(length_vector, v, c=1.0):
    L, v = smart_array(length_vector), smart_array(v)
    v2 = smart_dot(v, v)
    if not is_symbolic(v2) and v2 == 0:
        return L
    L_parallel = (smart_dot(L, v) / v2) * v
    L_perp = L - L_parallel
    return L_perp + L_parallel / gamma(v, c)


def simultaneity_velocity(delta_t, delta_x, c=1.0):
    v = c**2 * delta_t / delta_x
    if not is_symbolic(v) and abs(v) >= c:
        raise ValueError("Para hacer simultáneos esos eventos se requeriría |v| >= c.")
    return v
