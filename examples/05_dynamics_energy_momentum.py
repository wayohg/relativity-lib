"""
Relativistic dynamics: energy, momentum and center-of-momentum velocity.

Run from project root:
    python examples/05_dynamics_energy_momentum.py
"""

from __future__ import annotations

from relativity.sr.dynamics import (
    total_energy,
    kinetic_energy,
    momentum,
    four_momentum,
    invariant_mass_from_energy_momentum,
    invariant_mass_system,
    center_of_momentum_velocity,
    transform_energy_momentum_1d,
    constant_force_1d,
)

from _helpers import print_header


def main() -> None:
    print_header("05 Dynamics: energy and momentum")

    c = 1.0
    m = 2.0
    v = [0.6 * c, 0.0, 0.0]

    E = total_energy(m, v, c=c)
    K = kinetic_energy(m, v, c=c)
    p = momentum(m, v, c=c)
    P4 = four_momentum(m, v, c=c)

    print("single particle")
    print("E:", E)
    print("K:", K)
    print("p:", p)
    print("P4:", P4)
    print("m invariant:", invariant_mass_from_energy_momentum(E, p, c=c))

    energies = [total_energy(1.0, [0.6, 0.0, 0.0], c=c), total_energy(1.0, [-0.3, 0.0, 0.0], c=c)]
    momenta = [momentum(1.0, [0.6, 0.0, 0.0], c=c), momentum(1.0, [-0.3, 0.0, 0.0], c=c)]

    print("\ntwo-particle system")
    print("invariant system mass:", invariant_mass_system(energies, momenta, c=c))
    print("center-of-momentum velocity:", center_of_momentum_velocity(energies, momenta, c=c))

    print("\nenergy-momentum boost along x")
    Ep, pxp = transform_energy_momentum_1d(E, p[0], 0.2 * c, c=c)
    print("E' =", Ep)
    print("px' =", pxp)

    print("\nconstant force in 1D")
    t, x, velocity, energy, mom = constant_force_1d(t=3.0, force=0.5, mass=1.0, v0=0.0, x0=0.0, c=c)
    print("t:", t)
    print("x:", x)
    print("v:", velocity)
    print("E:", energy)
    print("p:", mom)


if __name__ == "__main__":
    main()
