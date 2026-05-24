import numpy as np

from relativity.math.tensors import (
    MinkowskiMetric
)

from relativity.utils import (
    smart_dot,
    smart_sqrt,
    is_symbolic,
    smart_array
)


eta = MinkowskiMetric().components


def minkowski_dot(a, b):

    a = smart_array(a)
    b = smart_array(b)

    return smart_dot(
        a,
        eta @ b
    )


def spacetime_interval(x):

    return minkowski_dot(x, x)


def classify_interval(x):

    s2 = spacetime_interval(x)

    if is_symbolic(s2):
        return s2

    if s2 > 0:
        return "timelike"

    elif s2 < 0:
        return "spacelike"

    return "lightlike"


def proper_time(dx, c):

    s2 = spacetime_interval(dx)

    return smart_sqrt(s2) / c


def lower_index(v):

    return eta @ smart_array(v)


def raise_index(v):

    return eta @ smart_array(v)