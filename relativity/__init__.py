"""
Relativity: A relativistic spacetime engine based on 4-vectors.

Modules
-------
physics : FourVector, Event, ReferenceFrame, FourWaveVector, Worldline, Particle, Photon
math    : Lorentz boosts, Minkowski metric, tensors
sr      : Collision
"""

# Physics layer
from .physics import FourVector, Event, ReferenceFrame, FourWaveVector, Worldline
from .physics.particle import Particle
from .physics.photon import Photon

# Math utilities
from .math import (
    minkowski_dot,
    spacetime_interval,
    interval_squared,
    classify_interval,
    proper_time,
    lower_index,
    raise_index,
    gamma,
    boost_matrix,
    relativistic_velocity_addition,
    time_dilation,
    length_contraction,
    Tensor,
    MinkowskiMetric,
    ElectromagneticTensor,
)

# SR processes
from .sr import Collision

# Constants
from .constants import C, PLANCK, ELECTRON_VOLT, KEV, MEV, GEV, TEV, YEAR, LIGHTYEAR

# Utilities
from .utils import (
    smart_array,
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
    pprint,
    is_symbolic,
)

__all__ = [
    # Physics
    "FourVector", "Event", "ReferenceFrame", "FourWaveVector",
    "Worldline", "Particle", "Photon",
    # Math
    "minkowski_dot", "spacetime_interval", "interval_squared",
    "classify_interval", "proper_time", "lower_index", "raise_index",
    "gamma", "boost_matrix", "relativistic_velocity_addition",
    "time_dilation", "length_contraction",
    "Tensor", "MinkowskiMetric", "ElectromagneticTensor",
    # SR
    "Collision",
    # Constants
    "C", "PLANCK", "ELECTRON_VOLT", "KEV", "MEV", "GEV", "TEV",
    "YEAR", "LIGHTYEAR",
    # Utils
    "smart_array", "smart_sqrt", "smart_exp", "smart_sin", "smart_cos",
    "smart_tanh", "smart_acosh", "smart_dot", "smart_norm", "smart_cross",
    "gamma_factor", "beta_factor", "rapidity_from_velocity",
    "velocity_from_rapidity", "simplify", "expand", "factor",
    "smart_equal", "smart_inverse", "smart_det", "pprint", "is_symbolic",
]
