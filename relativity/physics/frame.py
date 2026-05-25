"""Inertial reference frames."""
from __future__ import annotations
from relativity.physics.fourvector import FourVector
from relativity.math.lorentz import boost_matrix, relativistic_velocity_addition
from relativity.utils import smart_array, gamma_factor


class ReferenceFrame:
    def __init__(self, name="S", velocity=None, relative_to=None, c=1.0):
        self.name = name
        self.v = smart_array(velocity if velocity is not None else [0, 0, 0])
        self.relative_to = relative_to
        self.c = c

    @property
    def velocity(self): return self.v
    @property
    def beta(self): return self.v / self.c

    def velocity_wrt_root(self):
        if self.relative_to is None:
            return self.v
        return relativistic_velocity_addition(self.v, self.relative_to.velocity_wrt_root(), self.c)

    def velocity_wrt(self, other):
        """Velocity of this frame relative to other frame."""
        return relativistic_velocity_addition(self.velocity_wrt_root(), -other.velocity_wrt_root(), self.c)

    def transformation_to(self, other):
        """Matrix that transforms coordinates measured in this frame into other frame."""
        return boost_matrix(other.velocity_wrt(self), self.c)

    def transform_fourvector(self, fourvector, other):
        vec = fourvector.vec if hasattr(fourvector, "vec") else smart_array(fourvector)
        return FourVector(*(self.transformation_to(other) @ vec))

    def four_velocity(self):
        g = gamma_factor(self.v, self.c)
        return FourVector(g * self.c, *(g * self.v))

    def __repr__(self):
        return f"ReferenceFrame(name={self.name!r}, velocity={self.v})"
