"""Lorentz transformations with numeric/symbolic hybrid support."""

from __future__ import annotations

import numpy as np

from relativity.utils import (
    smart_array,
    smart_dot,
    smart_sqrt,
    smart_outer,
    smart_eye,
    smart_matmul,
    is_symbolic,
    simplify,
)


def gamma(v, c):
    """Lorentz factor gamma = 1/sqrt(1 - |v|²/c²)."""
    v = smart_array(v)
    beta2 = smart_dot(v, v) / c**2
    if not is_symbolic(beta2) and beta2 >= 1:
        raise ValueError("Velocidad superlumínica no permitida.")
    return simplify(1 / smart_sqrt(1 - beta2))


def boost_matrix(v, c):
    """
    4x4 Lorentz boost matrix for coordinates X=(ct,x,y,z).

    Sign convention: this maps from S to S' moving with velocity +v
    relative to S, so ct' = gamma(ct - beta·r).
    """
    v = smart_array(v)
    v2 = smart_dot(v, v)
    symbolic = is_symbolic(v) or is_symbolic(c) or is_symbolic(v2)

    if not symbolic and v2 == 0:
        return np.eye(4)

    g = gamma(v, c)
    beta = v / c
    beta2 = smart_dot(beta, beta)

    Lambda = smart_eye(4, symbolic=symbolic)
    Lambda[0, 0] = g
    Lambda[0, 1:] = -g * beta
    Lambda[1:, 0] = -g * beta

    if not symbolic and beta2 == 0:
        return Lambda

    Lambda[1:, 1:] = Lambda[1:, 1:] + (g - 1) * smart_outer(beta, beta) / beta2
    return simplify(Lambda)


def lorentz_transform_fourvector(X, v, c):
    X = smart_array(X)
    return simplify(smart_matmul(boost_matrix(v, c), X))


def inverse_lorentz_transform_fourvector(Xprime, v, c):
    return lorentz_transform_fourvector(Xprime, -smart_array(v), c)


def relativistic_velocity_addition(u, v, c):
    """
    Transform object velocity u from S to S' where S' moves at +v w.r.t. S.

    For 1D this reduces to u' = (u - v)/(1 - uv/c²).
    """
    u = smart_array(u)
    v = smart_array(v)
    v2 = smart_dot(v, v)

    if not is_symbolic(v2) and v2 == 0:
        return u

    uv_dot = smart_dot(u, v)
    g = gamma(v, c)
    factor = 1 - uv_dot / c**2

    u_parallel = smart_dot(u, v) / v2 * v
    u_perp = u - u_parallel

    result_parallel = (u_parallel - v) / factor
    result_perp = u_perp / (g * factor)
    return simplify(result_parallel + result_perp)


def velocity_addition_1d(u, v, c):
    return simplify((u - v) / (1 - u * v / c**2))


def inverse_velocity_addition_1d(u_prime, v, c):
    return simplify((u_prime + v) / (1 + u_prime * v / c**2))


def simultaneity_velocity(delta_t, delta_r, c):
    """1D boost velocity that makes two events simultaneous, if possible."""
    if not is_symbolic(delta_r) and delta_r == 0:
        raise ValueError("Eventos coincidentes espacialmente.")

    v = simplify(c**2 * delta_t / delta_r)

    if not is_symbolic(v) and abs(v) >= c:
        raise ValueError("Requiere velocidad superlumínica.")
    return v


def time_dilation(proper_time, v, c):
    return simplify(gamma(v, c) * proper_time)


def length_contraction(length_vector, v, c):
    """Contract only the component of length_vector parallel to v."""
    L = smart_array(length_vector)
    v = smart_array(v)
    v2 = smart_dot(v, v)

    if not is_symbolic(v2) and v2 == 0:
        return L

    g = gamma(v, c)
    L_parallel = smart_dot(L, v) / v2 * v
    L_perp = L - L_parallel
    return simplify(L_perp + L_parallel / g)
