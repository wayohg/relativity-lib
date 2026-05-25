"""
relativity
==========

Tools for special-relativistic calculations, simulations and visualization.

Recommended public API
----------------------
Use submodules for calculations to avoid name collisions:

    from relativity.sr import kinematics, dynamics, doppler, decay
    from relativity.physics import Event, Particle, Photon, Worldline
    from relativity.plotting import spacetime, worldline

Convenience imports are kept only for constants and core physical classes.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Constants
from .constants import C, PLANCK, ELECTRON_VOLT, KEV, MEV, GEV, TEV, YEAR, LIGHTYEAR

# Core physical objects
from .physics import (
    FourVector,
    Event,
    ReferenceFrame,
    Particle,
    Photon,
    FourWaveVector,
    Worldline,
)

# Subpackages. Import these as namespaces instead of flattening all functions.
from . import math
from . import physics
from . import sr
from . import plotting

__all__ = [
    "__version__",
    # constants
    "C",
    "PLANCK",
    "ELECTRON_VOLT",
    "KEV",
    "MEV",
    "GEV",
    "TEV",
    "YEAR",
    "LIGHTYEAR",
    # core objects
    "FourVector",
    "Event",
    "ReferenceFrame",
    "Particle",
    "Photon",
    "FourWaveVector",
    "Worldline",
    # subpackages
    "math",
    "physics",
    "sr",
    "plotting",
]
