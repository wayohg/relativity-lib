import numpy as np
from relativity.math.minkowski import interval_squared

class FourVector:
    def __init__(self, ct, x, y, z):
        self.vec = smart_array([ct, x, y, z], dtype=float)

    @classmethod
    def from_time_space(cls, t, r, c):
        return cls(c*t, r[0], r[1], r[2])

    @property
    def ct(self):
        return self.vec[0]

    @property
    def spatial(self):
        return self.vec[1:]

    def interval_squared(self):
        return interval_squared(self.vec)

    def __repr__(self):
        return f"FourVector({self.vec})"