import numpy as np

from relativity.physics.particle import Particle
from relativity.physics.photon import Photon

from relativity.constants import C


class Collision:

    def __init__(self, particles):

        self.particles = particles

    @property
    def total_energy(self):

        return sum(p.energy for p in self.particles)

    @property
    def total_momentum(self):

        return sum(
            (p.momentum for p in self.particles),
            start=np.zeros(3)
        )

    @property
    def total_four_momentum(self):

        total = np.zeros(4)

        for p in self.particles:

            total += p.four_momentum

        return total

    def invariant_mass(self):

        P = self.total_four_momentum

        E = P[0] * C

        p = P[1:]

        m2 = (E**2 / C**4) - (
            np.dot(p, p) / C**2
        )

        if m2 < 0:
            return 0

        return smart_sqrt(m2)

    def center_of_mass_velocity(self):

        E = self.total_energy

        p = self.total_momentum

        return (C**2 * p) / E

    def check_conservation(self, final_particles):

        initial_P = self.total_four_momentum

        final_P = np.zeros(4)

        for p in final_particles:

            final_P += p.four_momentum

        return np.allclose(initial_P, final_P)

    def electron_positron_annihilation(self):

        """
        Simplified symmetric annihilation.
        """

        E_total = self.total_energy

        photon_energy = E_total / 2

        h = 6.62607015e-34

        frequency = photon_energy / h

        photon1 = Photon(
            frequency=frequency,
            direction=[1,0,0]
        )

        photon2 = Photon(
            frequency=frequency,
            direction=[-1,0,0]
        )

        return photon1, photon2

    def __repr__(self):

        return (
            f"Collision(num_particles={len(self.particles)})"
        )