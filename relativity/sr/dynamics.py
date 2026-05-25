"""Energy-momentum relations for special relativity."""
from __future__ import annotations
from relativity.utils import smart_array, smart_dot, smart_sqrt, gamma_factor, simplify


def momentum(mass, velocity, c=1.0):
    velocity = smart_array(velocity)
    return gamma_factor(velocity, c) * mass * velocity


def energy(mass, velocity, c=1.0):
    return gamma_factor(smart_array(velocity), c) * mass * c**2


def kinetic_energy(mass, velocity, c=1.0):
    return energy(mass, velocity, c) - mass * c**2


def rest_energy(mass, c=1.0):
    return mass * c**2


def four_momentum(mass, velocity, c=1.0):
    p = momentum(mass, velocity, c)
    E = energy(mass, velocity, c)
    return smart_array([E / c, *p])


def invariant_mass_from_four_momentum(P, c=1.0):
    P = smart_array(P)
    E, p = P[0] * c, P[1:]
    return simplify(smart_sqrt(E**2 / c**4 - smart_dot(p, p) / c**2))


def energy_from_momentum(mass, p, c=1.0):
    p = smart_array(p)
    return smart_sqrt((mass * c**2)**2 + smart_dot(p, p) * c**2)


__all__ = ["momentum", "energy", "kinetic_energy", "rest_energy", "four_momentum", "invariant_mass_from_four_momentum", "energy_from_momentum"]
