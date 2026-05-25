"""
relativity.plotting
===================

Plotting utilities for special relativity.

This package exposes plotting modules as namespaces instead of importing every
plot function into one flat namespace. This keeps imports predictable and avoids
future collisions as the library grows.

Recommended usage:

    from relativity.plotting import kinematics, dynamics, decay, spacetime

    kinematics.plot_gamma_vs_beta(show=True)
    spacetime.plot_spacetime_diagram(...)

Direct imports from a concrete module are also supported:

    from relativity.plotting.kinematics import plot_gamma_vs_beta
"""

from __future__ import annotations

from . import style
from . import utils
from . import spacetime
from . import worldline
from . import kinematics
from . import dynamics
from . import decay

__all__ = [
    "style",
    "utils",
    "spacetime",
    "worldline",
    "kinematics",
    "dynamics",
    "decay",
]
