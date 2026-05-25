"""Relativistic collision / decay utilities."""

from __future__ import annotations

from relativity.physics.photon import Photon
from relativity.constants import C, PLANCK
from relativity.utils import smart_array, smart_dot, smart_sqrt, smart_equal, is_symbolic, simplify


class Collision:
    def __init__(self, particles, c=C):
        self.particles = list(particles)
        self.c = c

    @property
    def total_energy(self):
        return simplify(sum(p.energy for p in self.particles))

    @property
    def total_momentum(self):
        if not self.particles:
            return smart_array([0, 0, 0])
        total = smart_array([0, 0, 0], dtype=object if any(is_symbolic(p.momentum) for p in self.particles) else float)
        for p in self.particles:
            total = total + p.momentum
        return simplify(total)

    @property
    def total_four_momentum(self):
        if not self.particles:
            return smart_array([0, 0, 0, 0])
        total = smart_array([0, 0, 0, 0], dtype=object if any(is_symbolic(p.four_momentum) for p in self.particles) else float)
        for p in self.particles:
            total = total + p.four_momentum
        return simplify(total)

    def invariant_mass(self):
        P = self.total_four_momentum
        E = P[0] * self.c
        p = P[1:]
        m2 = simplify(E**2 / self.c**4 - smart_dot(p, p) / self.c**2)
        if not is_symbolic(m2) and m2 < 0:
            if m2 < -1e-10:
                raise ValueError(f"Unphysical negative m²={m2:.3e}. Check four-momenta consistency.")
            return 0.0
        return simplify(smart_sqrt(m2))

    def center_of_mass_velocity(self):
        E = self.total_energy
        if not is_symbolic(E) and E == 0:
            raise ValueError("Total energy is zero; cannot compute center-of-mass velocity.")
        return simplify((self.c**2 * self.total_momentum) / E)

    def check_conservation(self, final_particles, tol=1e-10):
        final = Collision(final_particles, c=self.c)
        return smart_equal(self.total_four_momentum, final.total_four_momentum, tol=tol)

    def electron_positron_annihilation(self):
        E_total = self.total_energy
        photon_energy = E_total / 2
        h = getattr(self.particles[0], "h", PLANCK) if self.particles else PLANCK
        frequency = photon_energy / h
        return (
            Photon(frequency=frequency, direction=[1, 0, 0], c=self.c, h=h),
            Photon(frequency=frequency, direction=[-1, 0, 0], c=self.c, h=h),
        )

    def __repr__(self):
        return f"Collision(num_particles={len(self.particles)})"
