"""Mathematical tools for special relativity."""

from __future__ import annotations

from . import lorentz
from . import minkowski
from . import tensors

from .lorentz import (
    boost_matrix,
    gamma,
    lorentz_transform_fourvector,
    inverse_lorentz_transform_fourvector,
    relativistic_velocity_addition,
    velocity_addition_1d,
    inverse_velocity_addition_1d,
    simultaneity_velocity,
    time_dilation,
    length_contraction,
)
from .minkowski import (
    minkowski_dot,
    spacetime_interval,
    interval_squared,
    classify_interval,
    proper_time,
    lower_index,
    raise_index,
    eta,
)
from .tensors import Tensor, MinkowskiMetric, ElectromagneticTensor

__all__ = [
    "lorentz",
    "minkowski",
    "tensors",
    "boost_matrix",
    "gamma",
    "lorentz_transform_fourvector",
    "inverse_lorentz_transform_fourvector",
    "relativistic_velocity_addition",
    "velocity_addition_1d",
    "inverse_velocity_addition_1d",
    "simultaneity_velocity",
    "time_dilation",
    "length_contraction",
    "minkowski_dot",
    "spacetime_interval",
    "interval_squared",
    "classify_interval",
    "proper_time",
    "lower_index",
    "raise_index",
    "eta",
    "Tensor",
    "MinkowskiMetric",
    "ElectromagneticTensor",
]
