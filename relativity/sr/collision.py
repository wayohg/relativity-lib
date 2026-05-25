"""Collision and decay bookkeeping using four-momentum conservation."""
from __future__ import annotations
import numpy as np
from relativity.constants import C, PLANCK
from relativity.physics.photon import Photon
from relativity.utils import smart_array, smart_dot, smart_sqrt, smart_equal, is_symbolic, simplify


class Collision:
    def __init__(self, particles, c=C):
        self.particles = list(particles)
        self.c = c

    @property
    def total_energy(self):
        return sum(p.energy for p in self.particles)

    @property
    def total_momentum(self):
        total = smart_array([0, 0, 0])
        for p in self.particles:
            total = total + p.momentum
        return total

    @property
    def total_four_momentum(self):
        total = smart_array([0, 0, 0, 0])
        for p in self.particles:
            total = total + p.four_momentum
        return total

    def invariant_mass(self):
        P = self.total_four_momentum
        E, p = P[0] * self.c, P[1:]
        m2 = E**2 / self.c**4 - smart_dot(p, p) / self.c**2
        if not is_symbolic(m2) and m2 < 0 and abs(m2) < 1e-12:
            m2 = 0
        return simplify(smart_sqrt(m2))

    def center_of_momentum_velocity(self):
        return (self.c**2 * self.total_momentum) / self.total_energy

    def check_conservation(self, final_particles, tol=1e-9):
        total = smart_array([0, 0, 0, 0])
        for p in final_particles:
            total = total + p.four_momentum
        return smart_equal(self.total_four_momentum, total, tol=tol)

    def two_photon_annihilation(self, axis=(1, 0, 0)):
        """Return two opposite photons in the center-of-momentum frame."""
        direction = smart_array(axis)
        E_photon = self.total_energy / 2
        f = E_photon / PLANCK
        return Photon(f, direction, c=self.c), Photon(f, -direction, c=self.c)

    def __repr__(self):
        return f"Collision(num_particles={len(self.particles)})"


def total_four_momentum(particles):
    return Collision(particles).total_four_momentum


def invariant_mass(particles, c=C):
    return Collision(particles, c=c).invariant_mass()


__all__ = ["Collision", "total_four_momentum", "invariant_mass"]
