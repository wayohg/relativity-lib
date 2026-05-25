"""
relativity.sr
=============

Special relativity solvers and utilities.

This package intentionally exposes modules as namespaces to avoid collisions
between common names such as gamma, beta, momentum, kinetic_energy, etc.

Recommended usage:

    from relativity.sr import kinematics, dynamics, doppler, decay

    kinematics.gamma(beta_value=0.8)
    dynamics.total_energy(mass=1.0, velocity=[0.8, 0, 0], c=1)

Direct imports from the concrete module are also encouraged:

    from relativity.sr.kinematics import gamma
    from relativity.sr.dynamics import total_energy
"""

from __future__ import annotations

from . import kinematics
from . import dynamics
from . import doppler
from . import decay
from . import collision

from .collision import Collision

__all__ = [
    "kinematics",
    "dynamics",
    "doppler",
    "decay",
    "collision",
    "Collision",
]
