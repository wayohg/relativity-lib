"""Four-vector object X^mu=(ct,x,y,z)."""

from __future__ import annotations

from relativity.utils import smart_array, simplify
from relativity.math.minkowski import spacetime_interval


class FourVector:
    def __init__(self, ct, x, y, z):
        self.vec = smart_array([ct, x, y, z])

    @classmethod
    def from_time_space(cls, t, r, c):
        r = smart_array(r)
        return cls(c * t, r[0], r[1], r[2])

    @property
    def ct(self):
        return self.vec[0]

    @property
    def spatial(self):
        return self.vec[1:]

    def interval_squared(self):
        return spacetime_interval(self.vec)

    def simplify(self):
        return FourVector(*simplify(self.vec))

    def __add__(self, other):
        return FourVector(*(self.vec + other.vec))

    def __sub__(self, other):
        return FourVector(*(self.vec - other.vec))

    def __mul__(self, scalar):
        return FourVector(*(self.vec * scalar))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self):
        return f"FourVector(ct={self.vec[0]}, x={self.vec[1]}, y={self.vec[2]}, z={self.vec[3]})"
