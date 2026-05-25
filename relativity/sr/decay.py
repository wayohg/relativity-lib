"""Relativistic lifetime and simple two-body decay helpers."""
from __future__ import annotations
from relativity.math.lorentz import gamma
from relativity.utils import smart_sqrt


def dilated_lifetime(proper_lifetime, velocity, c=1.0):
    return gamma(velocity, c) * proper_lifetime


def mean_decay_distance(proper_lifetime, velocity, c=1.0):
    import numpy as np
    vmag = smart_sqrt(sum(vi * vi for vi in velocity))
    return vmag * dilated_lifetime(proper_lifetime, velocity, c)


def two_body_decay_momentum(parent_mass, m1, m2, c=1.0):
    """Momentum magnitude of daughters in parent rest frame."""
    M = parent_mass
    term = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    return c * smart_sqrt(term) / (2 * M)


def two_body_decay_energies(parent_mass, m1, m2, c=1.0):
    M = parent_mass
    E1 = (M**2 + m1**2 - m2**2) * c**2 / (2 * M)
    E2 = (M**2 + m2**2 - m1**2) * c**2 / (2 * M)
    return E1, E2


__all__ = ["dilated_lifetime", "mean_decay_distance", "two_body_decay_momentum", "two_body_decay_energies"]
