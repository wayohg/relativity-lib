"""Four-vector class for x^mu=(ct,x,y,z)."""
from __future__ import annotations
from relativity.utils import smart_array
from relativity.math.minkowski import interval_squared, minkowski_dot, lower_index


class FourVector:
    def __init__(self, ct, x, y, z):
        self.vec = smart_array([ct, x, y, z])

    @classmethod
    def from_time_space(cls, t, r, c=1.0):
        r = smart_array(r)
        return cls(c * t, r[0], r[1], r[2])

    @property
    def ct(self): return self.vec[0]
    @property
    def spatial(self): return self.vec[1:]

    def interval_squared(self): return interval_squared(self.vec)
    def dot(self, other): return minkowski_dot(self.vec, other.vec if hasattr(other, "vec") else other)
    def lower(self): return lower_index(self.vec)

    def __add__(self, other):
        v = self.vec + (other.vec if hasattr(other, "vec") else smart_array(other))
        return FourVector(*v)

    def __sub__(self, other):
        v = self.vec - (other.vec if hasattr(other, "vec") else smart_array(other))
        return FourVector(*v)

    def __repr__(self):
        return f"FourVector(ct={self.ct}, spatial={self.spatial})"
