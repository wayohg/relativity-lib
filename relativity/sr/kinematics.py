"""Special-relativity kinematics: events, intervals, boosts, clocks and rods."""
from __future__ import annotations
from relativity.math.lorentz import gamma, boost_matrix, transform_fourvector, time_dilation, length_contraction, relativistic_velocity_addition, simultaneity_velocity
from relativity.math.minkowski import interval_squared, classify_interval, proper_time, proper_length


def lorentz_transform_event(t, r, v, c=1.0):
    """Return (t', r') for an event transformed by a boost velocity v."""
    x = [c * t, *r]
    xp = transform_fourvector(x, v, c)
    return xp[0] / c, xp[1:]


def twin_proper_time(coordinate_time, speed, c=1.0):
    """Proper time for inertial travel at constant speed during coordinate_time."""
    return coordinate_time / gamma([speed, 0, 0], c)


__all__ = [
    "gamma", "boost_matrix", "transform_fourvector", "lorentz_transform_event",
    "time_dilation", "length_contraction", "relativistic_velocity_addition",
    "simultaneity_velocity", "interval_squared", "classify_interval", "proper_time", "proper_length",
    "twin_proper_time",
]
