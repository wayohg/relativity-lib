"""
relativity.sr.dynamics
======================

High-level special-relativity dynamics utilities.

This module complements ``sr.kinematics``.  It focuses on energy,
momentum, forces, acceleration, power, work-energy relations, constant-force
motion, center-of-momentum quantities and simple threshold-energy problems.

Hybrid numeric/symbolic support
-------------------------------
The functions are written to work with ordinary floats/NumPy arrays and with
SymPy symbols/expressions.  For symbolic expressions, inequalities are not
forced unless they are directly decidable.

Conventions
-----------
Metric signature: (+---)
Four-momentum: P^mu = (E/c, px, py, pz)
SI units by default.
"""

from __future__ import annotations

import numpy as np

from relativity.constants import C
from relativity.utils import (
    smart_array,
    smart_dot,
    smart_norm,
    smart_sqrt,
    is_symbolic,
    simplify,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _as_vector3(v):
    """Return *v* as a 3-vector."""
    arr = smart_array(v)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-vector with shape (3,).")
    return arr


def _as_fourvector(P):
    """Return *P* as a 4-vector."""
    arr = smart_array(P)
    if arr.shape != (4,):
        raise ValueError("Expected a 4-vector with shape (4,).")
    return arr


def _is_numeric(x):
    return not is_symbolic(x)


def _check_massive_speed(v, c=C):
    """Validate |v| < c for numeric massive-particle velocities."""
    v = _as_vector3(v)
    if is_symbolic(v) or is_symbolic(c):
        return
    speed = float(np.linalg.norm(v.astype(float)))
    if speed >= float(c):
        raise ValueError(f"Massive-particle speed must satisfy |v| < c. Got |v|={speed:g}.")


def _check_nonzero_mass(mass):
    if _is_numeric(mass) and mass == 0:
        raise ValueError("This formula requires nonzero rest mass.")


def _split_parallel_perpendicular(vector, direction):
    """
    Split *vector* into components parallel and perpendicular to *direction*.
    """
    vector = _as_vector3(vector)
    direction = _as_vector3(direction)
    d2 = smart_dot(direction, direction)

    if _is_numeric(d2) and np.isclose(d2, 0):
        zero = smart_array([0, 0, 0])
        return zero, vector

    parallel = simplify((smart_dot(vector, direction) / d2) * direction)
    perpendicular = simplify(vector - parallel)
    return parallel, perpendicular


# ============================================================
# ENERGY, MOMENTUM AND MASS-SHELL RELATIONS
# ============================================================


def gamma(v, c=C):
    """Lorentz factor gamma(v) = 1/sqrt(1 - |v|^2/c^2)."""
    v = _as_vector3(v)
    _check_massive_speed(v, c)
    beta2 = smart_dot(v, v) / c**2
    return simplify(1 / smart_sqrt(1 - beta2))


def beta(v, c=C):
    """Dimensionless speed beta = |v|/c."""
    return simplify(smart_norm(_as_vector3(v)) / c)


def rest_energy(mass, c=C):
    """Rest energy E0 = m c^2."""
    return simplify(mass * c**2)


def total_energy(mass, v, c=C):
    """Total energy E = gamma m c^2."""
    return simplify(gamma(v, c=c) * mass * c**2)


relativistic_energy = total_energy


def kinetic_energy(mass, v, c=C):
    """Relativistic kinetic energy K = (gamma - 1) m c^2."""
    return simplify((gamma(v, c=c) - 1) * mass * c**2)


def momentum(mass, v, c=C):
    """Relativistic 3-momentum p = gamma m v."""
    v = _as_vector3(v)
    return simplify(gamma(v, c=c) * mass * v)


relativistic_momentum = momentum


def four_momentum(mass, v, c=C):
    """Four-momentum P^mu = (E/c, p_x, p_y, p_z)."""
    p = momentum(mass, v, c=c)
    E_over_c = total_energy(mass, v, c=c) / c
    return simplify(smart_array([E_over_c, *p]))


def energy_momentum_relation(mass, p, c=C):
    """
    Total energy from invariant mass and 3-momentum:

        E^2 = p^2 c^2 + m^2 c^4
    """
    p = _as_vector3(p)
    return simplify(smart_sqrt(smart_dot(p, p) * c**2 + mass**2 * c**4))


def kinetic_energy_from_momentum(mass, p, c=C):
    """Kinetic energy K = sqrt(p^2 c^2 + m^2 c^4) - m c^2."""
    return simplify(energy_momentum_relation(mass, p, c=c) - mass * c**2)


def gamma_from_energy(energy, mass, c=C):
    """gamma = E/(m c^2)."""
    _check_nonzero_mass(mass)
    return simplify(energy / (mass * c**2))


def beta_from_energy(energy, mass, c=C):
    """beta from total energy and rest mass."""
    g = gamma_from_energy(energy, mass, c=c)
    if _is_numeric(g) and g < 1:
        raise ValueError("Total energy must satisfy E >= m c^2.")
    return simplify(smart_sqrt(1 - 1 / g**2))


def speed_from_energy(energy, mass, c=C):
    """Speed magnitude from total energy and rest mass."""
    return simplify(c * beta_from_energy(energy, mass, c=c))


def momentum_magnitude_from_energy(energy, mass, c=C):
    """Momentum magnitude from total energy and rest mass."""
    value = energy**2 / c**2 - mass**2 * c**2
    if _is_numeric(value) and value < 0:
        if value > -1e-12:
            value = 0.0
        else:
            raise ValueError("Energy is below rest energy; momentum would be imaginary.")
    return simplify(smart_sqrt(value))


def velocity_from_momentum(mass, p, c=C):
    """Velocity vector from rest mass and 3-momentum: v = p c^2 / E."""
    p = _as_vector3(p)
    E = energy_momentum_relation(mass, p, c=c)
    return simplify(p * c**2 / E)


def invariant_mass_from_energy_momentum(energy, p, c=C):
    """Invariant mass from E and p: m^2 = E^2/c^4 - |p|^2/c^2."""
    p = _as_vector3(p)
    m2 = simplify(energy**2 / c**4 - smart_dot(p, p) / c**2)
    if _is_numeric(m2) and m2 < 0:
        if m2 > -1e-12:
            m2 = 0.0
        else:
            raise ValueError(f"Negative invariant mass squared: {m2:g}")
    return simplify(smart_sqrt(m2))


def invariant_mass_from_four_momentum(P, c=C):
    """Invariant mass from P^mu = (E/c, p_x, p_y, p_z)."""
    P = _as_fourvector(P)
    return invariant_mass_from_energy_momentum(P[0] * c, P[1:], c=c)


# ============================================================
# SYSTEMS OF PARTICLES
# ============================================================


def total_system_energy(particles_or_energies):
    """
    Sum energies.

    Accepts either a list of Particle-like objects with an ``energy`` property
    or a list of energy values.
    """
    total = 0
    for item in particles_or_energies:
        total += getattr(item, "energy", item)
    return simplify(total)


def total_system_momentum(particles_or_momenta):
    """
    Sum 3-momenta.

    Accepts either Particle-like objects with a ``momentum`` property or
    explicit 3-momentum vectors.
    """
    items = list(particles_or_momenta)
    if not items:
        return smart_array([0, 0, 0])

    symbolic = any(is_symbolic(getattr(item, "momentum", item)) for item in items)
    total = smart_array([0, 0, 0], dtype=object if symbolic else float)

    for item in items:
        total = total + _as_vector3(getattr(item, "momentum", item))

    return simplify(total)


def invariant_mass_system(energies, momenta, c=C):
    """
    Invariant mass of a multi-particle system from lists of energies and momenta.
    """
    E = total_system_energy(energies)
    p = total_system_momentum(momenta)
    return invariant_mass_from_energy_momentum(E, p, c=c)


def center_of_momentum_velocity(energies, momenta, c=C):
    """
    Center-of-momentum frame velocity:

        V_cm = c^2 P_total / E_total
    """
    E = total_system_energy(energies)
    if _is_numeric(E) and E == 0:
        raise ValueError("Total energy is zero; cannot compute center-of-momentum velocity.")
    p = total_system_momentum(momenta)
    return simplify(c**2 * p / E)


center_of_mass_velocity = center_of_momentum_velocity


def system_from_particles(particles, c=C):
    """
    Return a dictionary with total E, total p, invariant mass and COM velocity.
    """
    particles = list(particles)
    energies = [p.energy for p in particles]
    momenta = [p.momentum for p in particles]
    E = total_system_energy(energies)
    p = total_system_momentum(momenta)
    return {
        "energy": E,
        "momentum": p,
        "invariant_mass": invariant_mass_from_energy_momentum(E, p, c=c),
        "center_of_momentum_velocity": center_of_momentum_velocity(energies, momenta, c=c),
    }


# ============================================================
# ENERGY-MOMENTUM LORENTZ TRANSFORMS
# ============================================================


def transform_energy_momentum_1d(energy, px, v, c=C):
    """
    1D Lorentz transform of energy and x-momentum into a frame moving at v.

        E'  = gamma (E - v p_x)
        p'x = gamma (p_x - v E/c^2)
    """
    b = v / c
    if _is_numeric(b) and abs(b) >= 1:
        raise ValueError("Boost speed must satisfy |v| < c.")
    g = simplify(1 / smart_sqrt(1 - b**2))
    E_prime = g * (energy - v * px)
    px_prime = g * (px - v * energy / c**2)
    return simplify(E_prime), simplify(px_prime)


def inverse_transform_energy_momentum_1d(energy_prime, px_prime, v, c=C):
    """Inverse 1D energy-momentum transform, obtained with velocity -v."""
    return transform_energy_momentum_1d(energy_prime, px_prime, -v, c=c)


def transform_four_momentum_1d(P, v, c=C):
    """
    Transform P^mu=(E/c, px, py, pz) for a boost along x.
    """
    P = _as_fourvector(P)
    E_prime, px_prime = transform_energy_momentum_1d(P[0] * c, P[1], v, c=c)
    return simplify(smart_array([E_prime / c, px_prime, P[2], P[3]]))


# ============================================================
# FORCE, POWER AND ACCELERATION
# ============================================================


def power(force, velocity):
    """Instantaneous power delivered by a 3-force: P = F · v."""
    return simplify(smart_dot(_as_vector3(force), _as_vector3(velocity)))


def force_from_acceleration(mass, velocity, acceleration, c=C):
    """
    Relativistic 3-force from coordinate acceleration:

        F = m [gamma a + gamma^3 (v·a/c^2) v]

    where F = dp/dt.
    """
    _check_nonzero_mass(mass)
    v = _as_vector3(velocity)
    a = _as_vector3(acceleration)
    _check_massive_speed(v, c)
    g = gamma(v, c=c)
    return simplify(mass * (g * a + g**3 * smart_dot(v, a) * v / c**2))


def acceleration_from_force(mass, velocity, force, c=C):
    """
    Coordinate acceleration from 3-force.

    Components parallel to v obey a_parallel = F_parallel/(gamma^3 m),
    while perpendicular components obey a_perp = F_perp/(gamma m).
    """
    _check_nonzero_mass(mass)
    v = _as_vector3(velocity)
    F = _as_vector3(force)
    _check_massive_speed(v, c)
    g = gamma(v, c=c)
    F_parallel, F_perp = _split_parallel_perpendicular(F, v)
    return simplify(F_parallel / (g**3 * mass) + F_perp / (g * mass))


def longitudinal_acceleration_from_force(mass, speed_value, force_parallel, c=C):
    """1D acceleration parallel to motion: a_parallel = F_parallel/(gamma^3 m)."""
    _check_nonzero_mass(mass)
    g = simplify(1 / smart_sqrt(1 - speed_value**2 / c**2))
    return simplify(force_parallel / (g**3 * mass))


def transverse_acceleration_from_force(mass, speed_value, force_perpendicular, c=C):
    """Transverse acceleration: a_perp = F_perp/(gamma m)."""
    _check_nonzero_mass(mass)
    g = simplify(1 / smart_sqrt(1 - speed_value**2 / c**2))
    return simplify(force_perpendicular / (g * mass))


def proper_acceleration_1d(coordinate_acceleration, speed_value, c=C):
    """Proper acceleration for collinear 1D motion: alpha = gamma^3 a."""
    g = simplify(1 / smart_sqrt(1 - speed_value**2 / c**2))
    return simplify(g**3 * coordinate_acceleration)


def coordinate_acceleration_from_proper_1d(proper_acceleration, speed_value, c=C):
    """Coordinate acceleration for 1D motion: a = alpha/gamma^3."""
    g = simplify(1 / smart_sqrt(1 - speed_value**2 / c**2))
    return simplify(proper_acceleration / g**3)


def work_energy_delta(mass, v_initial, v_final, c=C):
    """Work done equals the change in kinetic energy."""
    return simplify(kinetic_energy(mass, v_final, c=c) - kinetic_energy(mass, v_initial, c=c))


# ============================================================
# CONSTANT FORCE / CONSTANT PROPER ACCELERATION MOTION
# ============================================================


def constant_force_1d(t, force, mass, v0=0, x0=0, c=C):
    """
    Exact 1D motion under constant lab-frame force F = dp/dt.

    Returns a dictionary with p(t), E(t), v(t), gamma(t), x(t).
    The initial momentum is computed from v0.
    """
    _check_nonzero_mass(mass)
    if _is_numeric(v0) and abs(v0) >= c:
        raise ValueError("Initial speed must satisfy |v0| < c.")

    g0 = simplify(1 / smart_sqrt(1 - v0**2 / c**2))
    p0 = simplify(g0 * mass * v0)
    p_t = simplify(p0 + force * t)
    E_t = simplify(smart_sqrt(mass**2 * c**4 + p_t**2 * c**2))
    v_t = simplify(p_t * c**2 / E_t)
    gamma_t = simplify(E_t / (mass * c**2))

    # Integral of v dt using dp = F dt.
    if _is_numeric(force) and np.isclose(force, 0):
        x_t = simplify(x0 + v0 * t)
    else:
        a = mass * c
        x_t = simplify(x0 + (c / force) * (smart_sqrt(a**2 + p_t**2) - smart_sqrt(a**2 + p0**2)))

    return {
        "momentum": p_t,
        "energy": E_t,
        "velocity": v_t,
        "gamma": gamma_t,
        "position": x_t,
    }


def constant_proper_acceleration_1d(tau, alpha, x0=0, t0=0, c=C):
    """
    Worldline for constant proper acceleration alpha, parameterized by proper time tau.

    Assumes motion starts from rest at tau=0:
        t(tau) = t0 + c/alpha sinh(alpha tau/c)
        x(tau) = x0 + c^2/alpha [cosh(alpha tau/c) - 1]
        v(tau) = c tanh(alpha tau/c)
    """
    import sympy as sp

    arg = alpha * tau / c
    if is_symbolic(arg):
        sinh = sp.sinh
        cosh = sp.cosh
        tanh = sp.tanh
    else:
        sinh = np.sinh
        cosh = np.cosh
        tanh = np.tanh

    if _is_numeric(alpha) and np.isclose(alpha, 0):
        return {"t": t0 + tau, "x": x0, "velocity": 0, "gamma": 1}

    t = simplify(t0 + (c / alpha) * sinh(arg))
    x = simplify(x0 + (c**2 / alpha) * (cosh(arg) - 1))
    v = simplify(c * tanh(arg))
    g = simplify(cosh(arg))
    return {"t": t, "x": x, "velocity": v, "gamma": g}


# ============================================================
# THRESHOLD ENERGY PROBLEMS
# ============================================================


def threshold_projectile_total_energy_lab(projectile_mass, target_mass, final_masses, c=C):
    """
    Threshold total energy of a projectile hitting a stationary target.

    Reaction form:
        projectile + target(at rest) -> final particles

    The returned energy includes the projectile rest energy.
    """
    M_final = sum(final_masses)
    numerator = (M_final**2 - projectile_mass**2 - target_mass**2) * c**2
    denominator = 2 * target_mass

    if _is_numeric(target_mass) and target_mass == 0:
        raise ValueError("Stationary target mass must be nonzero.")

    return simplify(numerator / denominator)


def threshold_projectile_kinetic_energy_lab(projectile_mass, target_mass, final_masses, c=C):
    """Threshold projectile kinetic energy for a stationary-target reaction."""
    E_total = threshold_projectile_total_energy_lab(projectile_mass, target_mass, final_masses, c=c)
    return simplify(E_total - projectile_mass * c**2)


# ============================================================
# CONVENIENCE EXPORTS
# ============================================================


__all__ = [
    "gamma",
    "beta",
    "rest_energy",
    "total_energy",
    "relativistic_energy",
    "kinetic_energy",
    "momentum",
    "relativistic_momentum",
    "four_momentum",
    "energy_momentum_relation",
    "kinetic_energy_from_momentum",
    "gamma_from_energy",
    "beta_from_energy",
    "speed_from_energy",
    "momentum_magnitude_from_energy",
    "velocity_from_momentum",
    "invariant_mass_from_energy_momentum",
    "invariant_mass_from_four_momentum",
    "total_system_energy",
    "total_system_momentum",
    "invariant_mass_system",
    "center_of_momentum_velocity",
    "center_of_mass_velocity",
    "system_from_particles",
    "transform_energy_momentum_1d",
    "inverse_transform_energy_momentum_1d",
    "transform_four_momentum_1d",
    "power",
    "force_from_acceleration",
    "acceleration_from_force",
    "longitudinal_acceleration_from_force",
    "transverse_acceleration_from_force",
    "proper_acceleration_1d",
    "coordinate_acceleration_from_proper_1d",
    "work_energy_delta",
    "constant_force_1d",
    "constant_proper_acceleration_1d",
    "threshold_projectile_total_energy_lab",
    "threshold_projectile_kinetic_energy_lab",
]
