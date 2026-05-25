"""
Symbolic calculations with SymPy.

Run from project root:
    python examples/02_symbolic_kinematics.py
"""

from __future__ import annotations

import sympy as sp

from relativity.sr.kinematics import (
    gamma,
    energy,
    momentum,
    lorentz_transform_event_1d,
    velocity_addition_1d,
)
from relativity.math.lorentz import boost_matrix

from _helpers import print_header


def main() -> None:
    print_header("02 Symbolic kinematics")

    m, v, u, c = sp.symbols("m v u c", positive=True)
    t, x = sp.symbols("t x", real=True)

    print("gamma(v):")
    print(gamma([v, 0, 0], c=c))

    print("\nenergy(m, v):")
    print(energy(m, [v, 0, 0], c=c))

    print("\nmomentum(m, v):")
    print(momentum(m, [v, 0, 0], c=c))

    print("\nLorentz transform 1D:")
    tp, xp = lorentz_transform_event_1d(t, x, v, c=c)
    print("t' =", tp)
    print("x' =", xp)

    print("\nVelocity addition 1D:")
    print(velocity_addition_1d(u, v, c=c))

    print("\nBoost matrix along x:")
    print(boost_matrix([v, 0, 0], c))


if __name__ == "__main__":
    main()
