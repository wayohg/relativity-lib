"""
relativity.sr.decay
===================

Special-relativity decay utilities.

This module is intended for common SR exercises involving unstable particles:
time dilation of lifetimes, decay lengths, survival probabilities, activity,
and two-body decays in the parent rest frame.

Hybrid numeric/symbolic support
-------------------------------
All public functions are written to work with floats/NumPy arrays and with
SymPy symbols/expressions.  Numeric domain checks are applied only when the
values can be compared safely; symbolic expressions are returned unevaluated
or simplified.

Conventions
-----------
Metric signature: (+---), SI units by default.
Masses are rest masses in kg unless you choose a consistent alternative unit
system.  If masses are entered in energy units with c=1, the same formulas
also work in natural units.
"""

from __future__ import annotations

import numpy as np

from relativity.constants import C
from relativity.utils import (
    smart_array,
    smart_dot,
    smart_norm,
    smart_sqrt,
    smart_exp,
    is_symbolic,
    simplify,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _is_numeric(x) -> bool:
    return not is_symbolic(x)


def _as_vector3(v):
    """Return *v* as a 3-vector."""
    arr = smart_array(v)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _speed_from_velocity(v):
    """Accept either a scalar speed or a 3-vector velocity."""
    arr = smart_array(v)

    if arr.shape == ():
        return arr.item()

    if arr.shape == (3,):
        return smart_norm(arr)

    raise ValueError("Expected a scalar speed or a 3-vector velocity.")


def _check_nonnegative(x, name: str):
    if _is_numeric(x) and x < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _check_positive(x, name: str):
    if _is_numeric(x) and x <= 0:
        raise ValueError(f"{name} must be positive.")


def _check_fraction(f, name: str = "fraction"):
    if _is_numeric(f) and not (0 <= f <= 1):
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1.")


def _check_speed(speed, c=C, allow_light: bool = False):
    if is_symbolic(speed) or is_symbolic(c):
        return

    speed_float = float(speed)
    c_float = float(c)

    if speed_float < 0:
        raise ValueError("Scalar speed cannot be negative.")

    if allow_light:
        if speed_float > c_float:
            raise ValueError("Speed cannot exceed c.")
    else:
        if speed_float >= c_float:
            raise ValueError("Massive-particle speed must satisfy speed < c.")


def _gamma_from_speed(speed, c=C):
    _check_speed(speed, c=c)
    return simplify(1 / smart_sqrt(1 - speed**2 / c**2))


def _lambda_kallen(x, y, z):
    """Kallen triangle function lambda(x,y,z)."""
    return simplify(x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z)


def _check_decay_allowed(parent_mass, daughter_mass_1, daughter_mass_2):
    if all(_is_numeric(x) for x in (parent_mass, daughter_mass_1, daughter_mass_2)):
        if parent_mass < daughter_mass_1 + daughter_mass_2:
            raise ValueError(
                "Two-body decay is not energetically allowed: "
                "parent_mass must satisfy M >= m1 + m2."
            )


# ============================================================
# BASIC RELATIVISTIC LIFETIME RELATIONS
# ============================================================


def gamma_from_velocity(velocity, c=C):
    """Lorentz factor from a scalar speed or a 3-vector velocity."""
    speed = _speed_from_velocity(velocity)
    return _gamma_from_speed(speed, c=c)


lorentz_factor = gamma_from_velocity


def beta_from_velocity(velocity, c=C):
    """Dimensionless speed beta = |v|/c from a scalar speed or velocity vector."""
    speed = _speed_from_velocity(velocity)
    _check_speed(speed, c=c)
    return simplify(speed / c)


def lab_lifetime(proper_lifetime, velocity, c=C):
    """Dilated mean lifetime tau_lab = gamma tau_0."""
    _check_positive(proper_lifetime, "proper_lifetime")
    return simplify(gamma_from_velocity(velocity, c=c) * proper_lifetime)


lifetime_lab = lab_lifetime
dilated_lifetime = lab_lifetime


def proper_lifetime_from_lab(lifetime_lab_value, velocity, c=C):
    """Recover proper lifetime tau_0 = tau_lab/gamma."""
    _check_positive(lifetime_lab_value, "lifetime_lab_value")
    return simplify(lifetime_lab_value / gamma_from_velocity(velocity, c=c))


def proper_time_elapsed(lab_time, velocity, c=C):
    """Elapsed proper time for a particle moving with constant velocity."""
    _check_nonnegative(lab_time, "lab_time")
    return simplify(lab_time / gamma_from_velocity(velocity, c=c))


def lab_time_from_proper_time(proper_time, velocity, c=C):
    """Elapsed lab time corresponding to a given proper time."""
    _check_nonnegative(proper_time, "proper_time")
    return simplify(gamma_from_velocity(velocity, c=c) * proper_time)


def decay_length(proper_lifetime, velocity, c=C):
    """
    Mean decay length in the lab frame.

        L = v tau_lab = beta gamma c tau_0
    """
    _check_positive(proper_lifetime, "proper_lifetime")
    speed = _speed_from_velocity(velocity)
    _check_speed(speed, c=c)
    return simplify(speed * lab_lifetime(proper_lifetime, speed, c=c))


mean_decay_length = decay_length


def proper_lifetime_from_decay_length(length, velocity, c=C):
    """Recover tau_0 from a measured mean decay length L."""
    _check_nonnegative(length, "length")
    speed = _speed_from_velocity(velocity)
    _check_speed(speed, c=c)
    if _is_numeric(speed) and speed == 0:
        raise ValueError("Velocity must be nonzero to infer lifetime from decay length.")
    return simplify(length / (speed * gamma_from_velocity(speed, c=c)))


def lab_time_from_distance(distance, velocity, c=C):
    """Lab travel time t = distance / speed."""
    _check_nonnegative(distance, "distance")
    speed = _speed_from_velocity(velocity)
    _check_speed(speed, c=c)
    if _is_numeric(speed) and speed == 0:
        raise ValueError("Velocity must be nonzero to compute travel time.")
    return simplify(distance / speed)


# ============================================================
# HALF-LIFE, EXPONENTIAL DECAY AND SURVIVAL PROBABILITY
# ============================================================


def mean_lifetime_from_half_life(half_life):
    """Mean lifetime tau = T_1/2 / ln(2)."""
    _check_positive(half_life, "half_life")
    if is_symbolic(half_life):
        import sympy as sp
        return simplify(half_life / sp.log(2))
    return half_life / np.log(2)


def half_life_from_mean_lifetime(mean_lifetime):
    """Half-life T_1/2 = tau ln(2)."""
    _check_positive(mean_lifetime, "mean_lifetime")
    if is_symbolic(mean_lifetime):
        import sympy as sp
        return simplify(mean_lifetime * sp.log(2))
    return mean_lifetime * np.log(2)


def lab_half_life(proper_half_life, velocity, c=C):
    """Dilated half-life in the lab frame."""
    _check_positive(proper_half_life, "proper_half_life")
    return simplify(gamma_from_velocity(velocity, c=c) * proper_half_life)


def survival_probability_time(time, proper_lifetime=None, velocity=None, *, lab_lifetime_value=None, c=C):
    """
    Probability that the particle has not decayed after lab time ``time``.

    Provide either:
        - proper_lifetime and velocity, or
        - lab_lifetime_value directly.
    """
    _check_nonnegative(time, "time")

    if lab_lifetime_value is None:
        if proper_lifetime is None or velocity is None:
            raise ValueError("Provide either lab_lifetime_value or both proper_lifetime and velocity.")
        lab_lifetime_value = lab_lifetime(proper_lifetime, velocity, c=c)

    _check_positive(lab_lifetime_value, "lab_lifetime_value")
    return simplify(smart_exp(-time / lab_lifetime_value))


def decay_probability_time(time, proper_lifetime=None, velocity=None, *, lab_lifetime_value=None, c=C):
    """Probability that the particle has decayed by lab time ``time``."""
    return simplify(1 - survival_probability_time(
        time,
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        lab_lifetime_value=lab_lifetime_value,
        c=c,
    ))


def survival_probability_distance(distance, proper_lifetime, velocity, c=C):
    """Survival probability after traveling a lab-frame distance."""
    _check_nonnegative(distance, "distance")
    L = decay_length(proper_lifetime, velocity, c=c)
    _check_positive(L, "decay_length")
    return simplify(smart_exp(-distance / L))


def decay_probability_distance(distance, proper_lifetime, velocity, c=C):
    """Probability that the particle decays within a lab-frame distance."""
    return simplify(1 - survival_probability_distance(distance, proper_lifetime, velocity, c=c))


def remaining_count(initial_count, time, proper_lifetime=None, velocity=None, *, lab_lifetime_value=None, c=C):
    """Expected number remaining after lab time ``time``."""
    _check_nonnegative(initial_count, "initial_count")
    return simplify(initial_count * survival_probability_time(
        time,
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        lab_lifetime_value=lab_lifetime_value,
        c=c,
    ))


def decayed_count(initial_count, time, proper_lifetime=None, velocity=None, *, lab_lifetime_value=None, c=C):
    """Expected number decayed by lab time ``time``."""
    return simplify(initial_count - remaining_count(
        initial_count,
        time,
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        lab_lifetime_value=lab_lifetime_value,
        c=c,
    ))


def activity(count, proper_lifetime=None, velocity=None, *, lab_lifetime_value=None, c=C):
    """
    Instantaneous decay rate/activity A = N/tau_lab.

    Units are decays per second when SI units are used.
    """
    _check_nonnegative(count, "count")

    if lab_lifetime_value is None:
        if proper_lifetime is None or velocity is None:
            raise ValueError("Provide either lab_lifetime_value or both proper_lifetime and velocity.")
        lab_lifetime_value = lab_lifetime(proper_lifetime, velocity, c=c)

    _check_positive(lab_lifetime_value, "lab_lifetime_value")
    return simplify(count / lab_lifetime_value)


# ============================================================
# INVERSE DECAY-LENGTH / SURVIVAL PROBLEMS
# ============================================================


def beta_gamma_from_decay_length(length, proper_lifetime, c=C):
    """Compute beta*gamma from mean decay length L = beta gamma c tau_0."""
    _check_nonnegative(length, "length")
    _check_positive(proper_lifetime, "proper_lifetime")
    return simplify(length / (c * proper_lifetime))


def beta_from_beta_gamma(beta_gamma):
    """Recover beta from beta*gamma."""
    _check_nonnegative(beta_gamma, "beta_gamma")
    return simplify(beta_gamma / smart_sqrt(1 + beta_gamma**2))


def gamma_from_beta_gamma(beta_gamma):
    """Recover gamma from beta*gamma."""
    _check_nonnegative(beta_gamma, "beta_gamma")
    return simplify(smart_sqrt(1 + beta_gamma**2))


def speed_from_decay_length(length, proper_lifetime, c=C):
    """Speed required to obtain a given mean decay length."""
    bg = beta_gamma_from_decay_length(length, proper_lifetime, c=c)
    return simplify(c * beta_from_beta_gamma(bg))


def speed_for_survival_fraction(distance, proper_lifetime, survival_fraction, c=C):
    """
    Speed required so that a fraction ``survival_fraction`` survives distance.

    Since f = exp[-L/(beta gamma c tau_0)], this gives
    beta gamma = distance / [-c tau_0 ln(f)].
    """
    _check_nonnegative(distance, "distance")
    _check_positive(proper_lifetime, "proper_lifetime")
    _check_fraction(survival_fraction, "survival_fraction")

    if _is_numeric(survival_fraction):
        if survival_fraction == 0:
            return 0.0
        if survival_fraction == 1:
            raise ValueError("Exact survival_fraction=1 requires speed -> c for nonzero distance.")

    if is_symbolic(survival_fraction):
        import sympy as sp
        denominator = -c * proper_lifetime * sp.log(survival_fraction)
    else:
        denominator = -c * proper_lifetime * np.log(survival_fraction)

    bg = simplify(distance / denominator)
    return simplify(c * beta_from_beta_gamma(bg))


# ============================================================
# TWO-BODY DECAY IN THE PARENT REST FRAME
# ============================================================


def q_value(parent_mass, daughter_masses, c=C):
    """Available rest-energy release Q = (M - sum(m_i)) c^2."""
    total_daughter_mass = sum(daughter_masses)
    Q = simplify((parent_mass - total_daughter_mass) * c**2)

    if _is_numeric(Q) and Q < 0:
        raise ValueError("Decay is energetically forbidden: Q < 0.")

    return Q


def two_body_decay_allowed(parent_mass, daughter_mass_1, daughter_mass_2):
    """Return True/False for numeric masses; symbolic cases return a condition."""
    if all(_is_numeric(x) for x in (parent_mass, daughter_mass_1, daughter_mass_2)):
        return parent_mass >= daughter_mass_1 + daughter_mass_2
    return simplify(parent_mass - daughter_mass_1 - daughter_mass_2)


def kallen_function(x, y, z):
    """Public Kallen triangle function lambda(x,y,z)."""
    return _lambda_kallen(x, y, z)


def two_body_decay_momentum(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """
    Magnitude of each daughter momentum in the parent rest frame.

        p* = c/(2M) sqrt[(M^2-(m1+m2)^2)(M^2-(m1-m2)^2)]

    With c=1 and masses in energy units, this returns momentum in the same
    natural unit system.
    """
    _check_positive(parent_mass, "parent_mass")
    _check_nonnegative(daughter_mass_1, "daughter_mass_1")
    _check_nonnegative(daughter_mass_2, "daughter_mass_2")
    _check_decay_allowed(parent_mass, daughter_mass_1, daughter_mass_2)

    radicand = simplify(
        (parent_mass**2 - (daughter_mass_1 + daughter_mass_2)**2)
        * (parent_mass**2 - (daughter_mass_1 - daughter_mass_2)**2)
    )

    if _is_numeric(radicand) and radicand < 0:
        if radicand > -1e-12:
            radicand = 0.0
        else:
            raise ValueError("Two-body momentum would be imaginary.")

    return simplify(c * smart_sqrt(radicand) / (2 * parent_mass))


def two_body_decay_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """
    Daughter total energies in the parent rest frame.

        E1* = (M^2 + m1^2 - m2^2)c^2/(2M)
        E2* = (M^2 + m2^2 - m1^2)c^2/(2M)
    """
    _check_positive(parent_mass, "parent_mass")
    _check_nonnegative(daughter_mass_1, "daughter_mass_1")
    _check_nonnegative(daughter_mass_2, "daughter_mass_2")
    _check_decay_allowed(parent_mass, daughter_mass_1, daughter_mass_2)

    E1 = (parent_mass**2 + daughter_mass_1**2 - daughter_mass_2**2) * c**2 / (2 * parent_mass)
    E2 = (parent_mass**2 + daughter_mass_2**2 - daughter_mass_1**2) * c**2 / (2 * parent_mass)
    return simplify(E1), simplify(E2)


def two_body_decay_kinetic_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """Daughter kinetic energies in the parent rest frame."""
    E1, E2 = two_body_decay_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    K1 = E1 - daughter_mass_1 * c**2
    K2 = E2 - daughter_mass_2 * c**2
    return simplify(K1), simplify(K2)


def two_body_decay_speeds(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """Daughter speeds in the parent rest frame."""
    p = two_body_decay_momentum(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    E1, E2 = two_body_decay_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=c)

    def speed(E, m):
        if _is_numeric(m) and m == 0:
            return c
        return simplify(p * c**2 / E)

    return speed(E1, daughter_mass_1), speed(E2, daughter_mass_2)


def two_body_decay_betas(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """Daughter beta values in the parent rest frame."""
    v1, v2 = two_body_decay_speeds(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    return simplify(v1 / c), simplify(v2 / c)


def two_body_decay_gammas(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """Daughter gamma values in the parent rest frame."""
    E1, E2 = two_body_decay_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=c)

    def gamma_from_E(E, m):
        if _is_numeric(m) and m == 0:
            return np.inf
        return simplify(E / (m * c**2))

    return gamma_from_E(E1, daughter_mass_1), gamma_from_E(E2, daughter_mass_2)


def two_body_decay_summary(parent_mass, daughter_mass_1, daughter_mass_2, c=C):
    """Convenience dictionary for a two-body decay in the parent rest frame."""
    p = two_body_decay_momentum(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    E1, E2 = two_body_decay_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    K1, K2 = two_body_decay_kinetic_energies(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    v1, v2 = two_body_decay_speeds(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    b1, b2 = simplify(v1 / c), simplify(v2 / c)

    return {
        "allowed": two_body_decay_allowed(parent_mass, daughter_mass_1, daughter_mass_2),
        "Q": q_value(parent_mass, [daughter_mass_1, daughter_mass_2], c=c),
        "momentum_magnitude": p,
        "energies": (E1, E2),
        "kinetic_energies": (K1, K2),
        "speeds": (v1, v2),
        "betas": (b1, b2),
    }


__all__ = [
    "gamma_from_velocity",
    "lorentz_factor",
    "beta_from_velocity",
    "lab_lifetime",
    "lifetime_lab",
    "dilated_lifetime",
    "proper_lifetime_from_lab",
    "proper_time_elapsed",
    "lab_time_from_proper_time",
    "decay_length",
    "mean_decay_length",
    "proper_lifetime_from_decay_length",
    "lab_time_from_distance",
    "mean_lifetime_from_half_life",
    "half_life_from_mean_lifetime",
    "lab_half_life",
    "survival_probability_time",
    "decay_probability_time",
    "survival_probability_distance",
    "decay_probability_distance",
    "remaining_count",
    "decayed_count",
    "activity",
    "beta_gamma_from_decay_length",
    "beta_from_beta_gamma",
    "gamma_from_beta_gamma",
    "speed_from_decay_length",
    "speed_for_survival_fraction",
    "q_value",
    "two_body_decay_allowed",
    "kallen_function",
    "two_body_decay_momentum",
    "two_body_decay_energies",
    "two_body_decay_kinetic_energies",
    "two_body_decay_speeds",
    "two_body_decay_betas",
    "two_body_decay_gammas",
    "two_body_decay_summary",
]
