"""
Relativity: Minimal relativistic spacetime engine based on 4-vectors.

Provides:
- FourVector
- Event
- ReferenceFrame
- Lorentz boost utilities
- Minkowski metric utilities
"""
"""
Main Library Init
Exposes core relativity classes and utilities.
"""

# Expose physics layer (importing from the sub-package directly)
from .physics import FourVector, Event, ReferenceFrame, FourWaveVector, Worldline

# Expose math utilities
from .math import minkowski_dot, interval_squared, gamma, boost_matrix

from relativity.constants import C,PLANCK, ELECTRON_VOLT, KEV, MEV, GEV, TEV, YEAR, LIGHTYEAR

from relativity.utils import (smart_array, 
                              smart_sqrt,
                              smart_exp,
                              smart_sin,
                              smart_cos,
                              smart_tanh,
                              smart_acosh,
                              smart_dot,
                              smart_norm,
                              smart_cross,
                              gamma_factor,
                              beta_factor,
                              rapidity_from_velocity,
                              velocity_from_rapidity,
                              simplify,
                              expand,
                              factor,
                              smart_equal,
                              smart_inverse,
                              smart_det,
                              pprint)

__all__ = [
    # Constants
    "C",
    "PLANCK",
    "ELECTRON_VOLT",
    "KEV",
    "MEV",
    "GEV",
    "TEV",
    "YEAR",
    "LIGHTYEAR",

    # Core physics
    "FourVector",
    "Event",
    "ReferenceFrame",
    "FourWaveVector",
    "Worldline",
    
    # Math utilities
    "minkowski_dot",
    "interval_squared",
    "gamma",
    "boost_matrix",

    "smart_array",
    "smart_sqrt",
    "smart_exp",
    "smart_sin",
    "smart_cos",
    "smart_tanh",
    "smart_acosh",
    "smart_dot",
    "smart_norm",
    "smart_cross",
    "gamma_factor",
    "beta_factor",
    "rapidity_from_velocity",
    "velocity_from_rapidity",
    "simplify",
    "expand",
    "factor",
    "smart_equal",
    "smart_inverse",
    "smart_det",
    "pprint"
]