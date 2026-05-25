"""
Particle and Photon objects.

Run from project root:
    python examples/03_particles_and_photons.py
"""

from __future__ import annotations

import numpy as np

from relativity.physics.particle import Particle
from relativity.physics.photon import Photon
from relativity.sr.dynamics import invariant_mass_from_energy_momentum

from _helpers import print_header


def main() -> None:
    print_header("03 Particles and photons")

    c = 1.0

    electron_like = Particle(
        mass=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.8 * c, 0.0, 0.0],
        name="massive particle",
        c=c,
    )

    print("Particle")
    print("speed:", electron_like.speed)
    print("beta:", electron_like.beta)
    print("gamma:", electron_like.gamma)
    print("energy:", electron_like.energy)
    print("momentum:", electron_like.momentum)
    print("four-momentum:", electron_like.four_momentum)
    print("invariant mass:", electron_like.invariant_mass)

    photon = Photon.from_wavelength(
        wavelength=2.0,
        direction=[1.0, 1.0, 0.0],
        name="diagonal photon",
        c=c,
        h=1.0,
    )

    print("\nPhoton")
    print("frequency:", photon.frequency)
    print("energy:", photon.energy)
    print("momentum:", photon.momentum)
    print("velocity:", photon.velocity)
    print("is lightlike:", photon.is_lightlike())

    # Independent invariant-mass check using E and p.
    m_inv = invariant_mass_from_energy_momentum(
        electron_like.energy,
        electron_like.momentum,
        c=c,
    )
    print("\nInvariant mass from sr.dynamics:", m_inv)

    # A simple sanity check useful while learning.
    assert np.isclose(float(electron_like.invariant_mass), 1.0)


if __name__ == "__main__":
    main()
