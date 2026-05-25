from .lorentz import boost_matrix, gamma
from .minkowski import (
    minkowski_dot,
    spacetime_interval,
    interval_squared,     # FIX: was missing from exports
    classify_interval,
    proper_time,
    lower_index,
    raise_index,
    eta,                  # FIX: export eta so event.py can import ETA from here
)
from .lorentz import (
    relativistic_velocity_addition,
    time_dilation,
    length_contraction,
)
from .tensors import Tensor, MinkowskiMetric, ElectromagneticTensor

__all__ = [
    "boost_matrix",
    "gamma",
    "minkowski_dot",
    "spacetime_interval",
    "interval_squared",
    "classify_interval",
    "proper_time",
    "lower_index",
    "raise_index",
    "eta",
    "relativistic_velocity_addition",
    "time_dilation",
    "length_contraction",
    "Tensor",
    "MinkowskiMetric",
    "ElectromagneticTensor",
]
