"""
relativity.sr.kinematics
========================

High-level special-relativity kinematics utilities.

This module is intended for solving textbook/homework problems and for
building simulations.  It works in a hybrid numeric/symbolic style:

- numeric values: NumPy arrays and floats
- symbolic values: SymPy symbols/expressions through relativity.utils

Conventions
-----------
Four-vectors use the (+---) metric and the coordinate order

    X^μ = (ct, x, y, z)
    U^μ = γ(c, vx, vy, vz)
    P^μ = (E/c, px, py, pz)

All SI units by default unless you pass a custom c.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from relativity.constants import C
from relativity.utils import (
    smart_array,
    smart_dot,
    smart_norm,
    smart_sqrt,
    smart_tanh,
    is_symbolic,
    simplify,
)
from relativity.math.minkowski import minkowski_dot, spacetime_interval
from relativity.math.lorentz import boost_matrix


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _as_vector3(v):
    """Return v as a 3-vector."""
    arr = smart_array(v)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _as_fourvector(x):
    """Return x as a 4-vector."""
    arr = smart_array(x)
    if arr.shape != (4,):
        raise ValueError("Expected a 4-vector with shape (4,).")
    return arr


def _is_numeric(x):
    return not is_symbolic(x)


def _check_subluminal(v, c=C, allow_equal=False):
    """
    Validate |v| < c for massive objects.

    For symbolic velocities no inequality is imposed automatically.
    """
    v = _as_vector3(v)

    if is_symbolic(v) or is_symbolic(c):
        return

    vmag = float(np.linalg.norm(v.astype(float)))
    limit = float(c)

    if allow_equal:
        bad = vmag > limit
        msg = "Speed cannot exceed c."
    else:
        bad = vmag >= limit
        msg = "Massive-particle speed must satisfy |v| < c."

    if bad:
        raise ValueError(f"{msg} Got |v|={vmag:g}, c={limit:g}.")


def _check_beta(beta, allow_equal=False):
    """Validate |beta| < 1 for numeric scalar beta."""
    if is_symbolic(beta):
        return

    b = float(abs(beta))
    if allow_equal:
        bad = b > 1.0
        msg = "|β| cannot exceed 1."
    else:
        bad = b >= 1.0
        msg = "Massive-particle β must satisfy |β| < 1."

    if bad:
        raise ValueError(f"{msg} Got β={beta:g}.")


# ============================================================
# BASIC FACTORS
# ============================================================

def speed(v):
    """Magnitude of a 3-velocity."""
    return simplify(smart_norm(_as_vector3(v)))


def beta(v, c=C):
    """Dimensionless speed β = |v|/c."""
    return simplify(speed(v) / c)


def beta_vector(v, c=C):
    """Dimensionless velocity vector β⃗ = v⃗/c."""
    return simplify(_as_vector3(v) / c)


def gamma(v=None, beta_value=None, c=C):
    """
    Lorentz factor γ.

    Use either:
        gamma(v=[vx, vy, vz], c=C)
    or:
        gamma(beta_value=b)
    """
    if beta_value is not None:
        _check_beta(beta_value)
        b2 = beta_value**2
    else:
        if v is None:
            raise ValueError("Pass either v or beta_value.")
        v = _as_vector3(v)
        _check_subluminal(v, c)
        b2 = smart_dot(v, v) / c**2

    return simplify(1 / smart_sqrt(1 - b2))


lorentz_factor = gamma


def beta_from_gamma(gamma_value):
    """β from γ: β = sqrt(1 - 1/γ²)."""
    if _is_numeric(gamma_value) and gamma_value < 1:
        raise ValueError("γ must be >= 1.")
    return simplify(smart_sqrt(1 - 1 / gamma_value**2))


def gamma_from_beta(beta_value):
    """γ from scalar β."""
    return gamma(beta_value=beta_value)


# ============================================================
# RAPIDITY AND PROPER VELOCITY
# ============================================================

def rapidity(v=None, beta_value=None, c=C):
    """
    Rapidity φ, with tanh(φ)=β.

    For a 3-vector velocity this returns the rapidity magnitude.
    """
    b = beta_value if beta_value is not None else beta(v, c)
    _check_beta(b)
    return simplify(sp.atanh(b) if is_symbolic(b) else np.arctanh(b))


def velocity_from_rapidity(phi, direction=None, c=C):
    """
    Convert rapidity to velocity.

    If direction is None, returns the signed 1D velocity v = c tanh(phi).
    If direction is a 3-vector, returns c tanh(phi) * unit(direction).
    """
    vmag = c * smart_tanh(phi)

    if direction is None:
        return simplify(vmag)

    n = _as_vector3(direction)
    n_norm = smart_norm(n)

    if _is_numeric(n_norm) and np.isclose(n_norm, 0):
        raise ValueError("Direction vector cannot be zero.")

    return simplify(vmag * n / n_norm)


def proper_velocity(v, c=C):
    """
    Proper velocity w⃗ = γ v⃗ = dx⃗/dτ.
    """
    v = _as_vector3(v)
    return simplify(gamma(v, c=c) * v)


# ============================================================
# FOUR-VELOCITY AND FOUR-MOMENTUM
# ============================================================

def four_velocity(v, c=C):
    """
    Four-velocity U^μ = γ(c, v⃗).

    Returns a 4-array: [γc, γvx, γvy, γvz].
    """
    v = _as_vector3(v)
    g = gamma(v, c=c)
    return simplify(smart_array([g * c, *(g * v)]))


def four_momentum(mass, v, c=C):
    """
    Massive-particle four-momentum P^μ = (E/c, p⃗).

    Returns a 4-array: [E/c, px, py, pz].
    """
    v = _as_vector3(v)
    g = gamma(v, c=c)
    p = g * mass * v
    E_over_c = g * mass * c
    return simplify(smart_array([E_over_c, *p]))


def energy(mass, v, c=C):
    """Total relativistic energy E = γmc²."""
    return simplify(gamma(v, c=c) * mass * c**2)


def rest_energy(mass, c=C):
    """Rest energy E0 = mc²."""
    return simplify(mass * c**2)


def kinetic_energy(mass, v, c=C):
    """Relativistic kinetic energy K = (γ - 1)mc²."""
    return simplify((gamma(v, c=c) - 1) * mass * c**2)


def momentum(mass, v, c=C):
    """Relativistic 3-momentum p⃗ = γm v⃗."""
    v = _as_vector3(v)
    return simplify(gamma(v, c=c) * mass * v)


def invariant_mass_from_four_momentum(P, c=C):
    """
    Invariant mass from P^μ = (E/c, p⃗).

    m² = (E/c²)² - |p⃗|²/c²
    """
    P = _as_fourvector(P)
    E = P[0] * c
    p = P[1:]
    m2 = E**2 / c**4 - smart_dot(p, p) / c**2

    if _is_numeric(m2) and m2 < 0:
        if m2 > -1e-12:
            m2 = 0.0
        else:
            raise ValueError(f"Negative invariant mass squared: {m2:g}")

    return simplify(smart_sqrt(m2))


def invariant_mass_from_energy_momentum(E, p, c=C):
    """
    Invariant mass from energy and 3-momentum.

    m² = E²/c⁴ - |p⃗|²/c²
    """
    p = _as_vector3(p)
    m2 = E**2 / c**4 - smart_dot(p, p) / c**2

    if _is_numeric(m2) and m2 < 0:
        if m2 > -1e-12:
            m2 = 0.0
        else:
            raise ValueError(f"Negative invariant mass squared: {m2:g}")

    return simplify(smart_sqrt(m2))


# ============================================================
# TIME DILATION, LENGTH CONTRACTION, SIMULTANEITY
# ============================================================

def dilated_time(proper_time, v, c=C):
    """
    Coordinate time measured in a frame where the clock moves at v.

    Δt = γ Δτ
    """
    return simplify(gamma(v, c=c) * proper_time)


def proper_time_from_lab_time(lab_time, v, c=C):
    """
    Proper time of a moving clock from lab-frame time.

    Δτ = Δt / γ
    """
    return simplify(lab_time / gamma(v, c=c))


def contracted_length(proper_length, v, c=C):
    """
    Length parallel to the motion measured in the lab.

    L = L0 / γ
    """
    return simplify(proper_length / gamma(v, c=c))


def proper_length_from_contracted(length, v, c=C):
    """Recover proper length L0 = γL."""
    return simplify(gamma(v, c=c) * length)


def relativity_of_simultaneity_time_offset(delta_x, v, c=C):
    """
    Time separation induced by changing frames for events simultaneous in S.

    For Δt=0 in S, the moving frame measures:
        Δt' = -γ v Δx / c²

    This 1D formula assumes motion along x.
    """
    vx = v[0] if hasattr(v, "__len__") else v
    b = vx / c
    _check_beta(b)
    g = gamma(beta_value=b)
    return simplify(-g * vx * delta_x / c**2)


def simultaneity_boost_velocity(delta_t, delta_x, c=C):
    """
    Velocity needed to make two 1D events simultaneous in S'.

    Uses Δt' = γ(Δt - vΔx/c²) = 0, hence:
        v = c² Δt / Δx
    """
    if _is_numeric(delta_x) and np.isclose(delta_x, 0):
        raise ValueError("delta_x cannot be zero.")

    v = c**2 * delta_t / delta_x

    if _is_numeric(v) and abs(v) >= c:
        raise ValueError("This would require |v| >= c.")

    return simplify(v)


# ============================================================
# LORENTZ TRANSFORMATIONS
# ============================================================

def lorentz_transform_fourvector(X, v, c=C):
    """
    Transform a four-vector X^μ=(ct,x,y,z) to a frame moving at v.

    This wraps relativity.math.lorentz.boost_matrix for numeric vectors.
    For symbolic 1D transformations, use lorentz_transform_event_1d.
    """
    X = _as_fourvector(X)
    v = _as_vector3(v)
    _check_subluminal(v, c, allow_equal=False)

    if is_symbolic(X) or is_symbolic(v) or is_symbolic(c):
        # General symbolic 3D boost matrices are possible, but the current
        # math.lorentz.boost_matrix is numeric. Keep this error explicit.
        raise TypeError(
            "General symbolic 3D Lorentz transforms are not supported here. "
            "Use lorentz_transform_event_1d for symbolic 1D boosts."
        )

    return boost_matrix(v, c) @ X


def inverse_lorentz_transform_fourvector(X_prime, v, c=C):
    """Inverse transform from S' to S by using velocity -v."""
    return lorentz_transform_fourvector(X_prime, -_as_vector3(v), c=c)


def lorentz_transform_event_1d(t, x, v, c=C):
    """
    1D Lorentz transform along x.

    Returns (t', x') with:
        t' = γ(t - vx/c²)
        x' = γ(x - vt)
    """
    b = v / c
    _check_beta(b)
    g = gamma(beta_value=b)
    t_p = g * (t - v * x / c**2)
    x_p = g * (x - v * t)
    return simplify(t_p), simplify(x_p)


def inverse_lorentz_transform_event_1d(t_prime, x_prime, v, c=C):
    """Inverse 1D Lorentz transform, obtained with velocity -v."""
    return lorentz_transform_event_1d(t_prime, x_prime, -v, c=c)


# ============================================================
# VELOCITY TRANSFORMATIONS
# ============================================================

def velocity_addition_1d(u, v, c=C):
    """
    Relativistic 1D velocity composition.

    If an object moves at u in S and S' moves at v relative to S:
        u' = (u - v)/(1 - uv/c²)

    This is the transformation of object velocity into the moving frame S'.
    """
    denom = 1 - u * v / c**2

    if _is_numeric(denom) and np.isclose(denom, 0):
        raise ZeroDivisionError("Velocity-addition denominator is zero.")

    result = (u - v) / denom

    if _is_numeric(result) and abs(result) > c * (1 + 1e-12):
        raise ValueError("Velocity transformation produced |u'| > c.")

    return simplify(result)


def inverse_velocity_addition_1d(u_prime, v, c=C):
    """
    Inverse 1D velocity transformation.

    u = (u' + v)/(1 + u'v/c²)
    """
    denom = 1 + u_prime * v / c**2

    if _is_numeric(denom) and np.isclose(denom, 0):
        raise ZeroDivisionError("Velocity-addition denominator is zero.")

    return simplify((u_prime + v) / denom)


def velocity_transform_3d(u, v, c=C):
    """
    Transform a 3-velocity u from S to S', where S' moves at v relative to S.

    Decomposes u into components parallel/perpendicular to v:

        u'_parallel = (u_parallel - v)/(1 - u·v/c²)
        u'_perp     = u_perp/(γ_v(1 - u·v/c²))

    Returns u' as a 3-vector.
    """
    u = _as_vector3(u)
    v = _as_vector3(v)

    _check_subluminal(u, c, allow_equal=True)
    _check_subluminal(v, c, allow_equal=False)

    if is_symbolic(u) or is_symbolic(v) or is_symbolic(c):
        raise TypeError(
            "Symbolic 3D velocity transforms are not supported. "
            "Use velocity_addition_1d for symbolic 1D cases."
        )

    v2 = smart_dot(v, v)

    if np.isclose(v2, 0):
        return u

    denom = 1 - smart_dot(u, v) / c**2

    if np.isclose(denom, 0):
        raise ZeroDivisionError("Velocity-transform denominator is zero.")

    u_parallel = (smart_dot(u, v) / v2) * v
    u_perp = u - u_parallel
    g_v = gamma(v, c=c)

    result = (u_parallel - v) / denom + u_perp / (g_v * denom)
    return simplify(result)


def inverse_velocity_transform_3d(u_prime, v, c=C):
    """
    Inverse 3D velocity transform from S' back to S.
    """
    return velocity_transform_3d(u_prime, -_as_vector3(v), c=c)


# ============================================================
# INTERVAL-BASED KINEMATICS
# ============================================================

def interval_between_events(t1, r1, t2, r2, c=C):
    """
    Minkowski interval squared between two events.

    s² = c²(t2 - t1)² - |r2 - r1|²
    """
    r1 = _as_vector3(r1)
    r2 = _as_vector3(r2)

    dX = smart_array([
        c * (t2 - t1),
        *(r2 - r1),
    ])

    return simplify(spacetime_interval(dX))


def proper_time_between_events(t1, r1, t2, r2, c=C):
    """
    Proper time between timelike-separated events.

    Δτ = sqrt(s²)/c, valid only for s² > 0.
    """
    s2 = interval_between_events(t1, r1, t2, r2, c=c)

    if _is_numeric(s2) and s2 <= 0:
        raise ValueError("Events are not timelike separated; proper time is not real.")

    return simplify(smart_sqrt(s2) / c)


def proper_distance_between_events(t1, r1, t2, r2, c=C):
    """
    Proper distance between spacelike-separated events.

    L0 = sqrt(-s²), valid only for s² < 0.
    """
    s2 = interval_between_events(t1, r1, t2, r2, c=c)

    if _is_numeric(s2) and s2 >= 0:
        raise ValueError("Events are not spacelike separated; proper distance is not real.")

    return simplify(smart_sqrt(-s2))


def classify_separation(t1, r1, t2, r2, c=C):
    """
    Classify event separation as 'timelike', 'spacelike', or 'lightlike'.

    For symbolic input returns the simplified interval expression instead.
    """
    s2 = interval_between_events(t1, r1, t2, r2, c=c)

    if is_symbolic(s2):
        return simplify(s2)

    if s2 > 0:
        return "timelike"
    if s2 < 0:
        return "spacelike"
    return "lightlike"


# ============================================================
# CONSTANT ACCELERATION, USEFUL FOR WORLDLINE EXERCISES
# ============================================================

def constant_proper_acceleration_worldline(tau, acceleration, c=C):
    """
    1D worldline for constant proper acceleration a.

    Returns (t(τ), x(τ)) assuming x(0)=c²/a and t(0)=0:

        ct = (c²/a) sinh(aτ/c)
        x  = (c²/a) cosh(aτ/c)

    Equivalently:
        t = (c/a) sinh(aτ/c)
        x = (c²/a) cosh(aτ/c)
    """
    a = acceleration

    if _is_numeric(a) and np.isclose(a, 0):
        raise ValueError("acceleration cannot be zero.")

    arg = a * tau / c

    if is_symbolic(arg):
        t = (c / a) * sp.sinh(arg)
        x = (c**2 / a) * sp.cosh(arg)
    else:
        t = (c / a) * np.sinh(arg)
        x = (c**2 / a) * np.cosh(arg)

    return simplify(t), simplify(x)


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "speed",
    "beta",
    "beta_vector",
    "gamma",
    "lorentz_factor",
    "beta_from_gamma",
    "gamma_from_beta",
    "rapidity",
    "velocity_from_rapidity",
    "proper_velocity",
    "four_velocity",
    "four_momentum",
    "energy",
    "rest_energy",
    "kinetic_energy",
    "momentum",
    "invariant_mass_from_four_momentum",
    "invariant_mass_from_energy_momentum",
    "dilated_time",
    "proper_time_from_lab_time",
    "contracted_length",
    "proper_length_from_contracted",
    "relativity_of_simultaneity_time_offset",
    "simultaneity_boost_velocity",
    "lorentz_transform_fourvector",
    "inverse_lorentz_transform_fourvector",
    "lorentz_transform_event_1d",
    "inverse_lorentz_transform_event_1d",
    "velocity_addition_1d",
    "inverse_velocity_addition_1d",
    "velocity_transform_3d",
    "inverse_velocity_transform_3d",
    "interval_between_events",
    "proper_time_between_events",
    "proper_distance_between_events",
    "classify_separation",
    "constant_proper_acceleration_worldline",
]
