import numpy as np

from relativity.physics.particle import Particle
from relativity.physics.photon import Photon
from relativity.constants import C, PLANCK


class Collision:
    """
    Relativistic collision / decay handler.
    Accepts any mix of Particle and Photon objects.
    """

    def __init__(self, particles, c=C):             # FIX: hardcoded C → configurable
        self.particles = particles
        self.c = c

    # =====================================================
    # CONSERVED QUANTITIES
    # =====================================================

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

    # =====================================================
    # INVARIANT MASS
    # =====================================================

    def invariant_mass(self):
        """
        Invariant mass of the system: M² = E²/c⁴ - |p|²/c².
        FIX: was using global C instead of self.c.
        """
        P = self.total_four_momentum
        E = P[0] * self.c
        p = P[1:]
        m2 = (E**2 / self.c**4) - np.dot(p, p) / self.c**2

        if m2 < 0:
            # FIX: small negative m2 (numerical noise) → 0; large negative is unphysical
            if m2 < -1e-10:
                raise ValueError(
                    f"Unphysical negative m²={m2:.3e}. "
                    "Check four-momenta consistency."
                )
            return 0.0

        return np.sqrt(m2)

    # =====================================================
    # CENTER OF MASS
    # =====================================================

    def center_of_mass_velocity(self):
        """
        Velocity of the center-of-mass frame: v_cm = c² p / E.
        FIX: was using global C instead of self.c.
        """
        E = self.total_energy
        p = self.total_momentum
        return (self.c**2 * p) / E

    # =====================================================
    # CONSERVATION CHECK
    # =====================================================

    def check_conservation(self, final_particles):
        """Return True if four-momentum is conserved (within tolerance)."""
        initial_P = self.total_four_momentum
        final_P = sum(
            (p.four_momentum for p in final_particles),
            start=np.zeros(4)
        )
        return np.allclose(initial_P, final_P, rtol=1e-8, atol=1e-12)

    # =====================================================
    # SPECIAL PROCESSES
    # =====================================================

    def electron_positron_annihilation(self):
        """
        Symmetric e⁺e⁻ → 2γ annihilation.
        Assumes equal-energy, back-to-back photons in the CM frame.
        FIX: now uses self.c and self.particles[0].h if available.
        """
        E_total = self.total_energy
        photon_energy = E_total / 2

        # Use h from a particle if it carries it, otherwise PLANCK
        h = getattr(self.particles[0], 'h', PLANCK) if self.particles else PLANCK
        frequency = photon_energy / h

        photon1 = Photon(frequency=frequency, direction=[1, 0, 0], c=self.c, h=h)
        photon2 = Photon(frequency=frequency, direction=[-1, 0, 0], c=self.c, h=h)

        return photon1, photon2

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        return f"Collision(num_particles={len(self.particles)})"
