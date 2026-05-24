import numpy as np
from relativity.utils import smart_array          # FIX: was missing; used below
from relativity.math.minkowski import spacetime_interval


class FourVector:
    def __init__(self, ct, x, y, z):
        self.vec = smart_array([ct, x, y, z], dtype=float)

    @classmethod
    def from_time_space(cls, t, r, c):
        r = smart_array(r, dtype=float)
        return cls(c * t, r[0], r[1], r[2])

    @property
    def ct(self):
        return self.vec[0]

    @property
    def spatial(self):
        return self.vec[1:]

    def interval_squared(self):
        return spacetime_interval(self.vec)

    # =====================================================
    # ALGEBRA  (FIX: missing operators)
    # =====================================================

    def __add__(self, other):
        v = self.vec + other.vec
        return FourVector(*v)

    def __sub__(self, other):
        v = self.vec - other.vec
        return FourVector(*v)

    def __mul__(self, scalar):
        v = self.vec * scalar
        return FourVector(*v)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self):
        return f"FourVector(ct={self.vec[0]:.4g}, x={self.vec[1]:.4g}, y={self.vec[2]:.4g}, z={self.vec[3]:.4g})"
