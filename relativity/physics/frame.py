"""Inertial reference frame."""

from __future__ import annotations

from relativity.physics.fourvector import FourVector
from relativity.math.lorentz import boost_matrix, relativistic_velocity_addition
from relativity.utils import smart_array, smart_dot, gamma_factor, is_symbolic, simplify, smart_matmul


class ReferenceFrame:
    def __init__(self, name, velocity, relative_to=None, c=1.0):
        self.name = name
        self.v = smart_array(velocity)
        self.c = c
        self.relative_to = relative_to

    @property
    def velocity(self):
        return self.v

    @property
    def beta(self):
        return simplify(self.v / self.c)

    def _velocity_wrt_root(self):
        if self.relative_to is None:
            return self.v
        parent_v = self.relative_to._velocity_wrt_root()
        # Velocity of this frame wrt root.
        return relativistic_velocity_addition(self.v, -parent_v, self.c)

    def velocity_wrt(self, other_frame):
        v_self = self._velocity_wrt_root()
        v_other = other_frame._velocity_wrt_root()
        return relativistic_velocity_addition(v_other, v_self, self.c)

    def _transformation_to(self, other):
        v_rel = self.velocity_wrt(other)
        return boost_matrix(v_rel, self.c)

    def _transform(self, fourvector, other):
        new_vec = smart_matmul(self._transformation_to(other), fourvector.vec)
        return FourVector(*new_vec)

    def four_velocity(self):
        v2 = smart_dot(self.v, self.v)
        if not is_symbolic(v2) and v2 == 0:
            return FourVector(self.c, 0, 0, 0)
        g = gamma_factor(self.v, self.c)
        return FourVector(g * self.c, *(g * self.v))

    def __repr__(self):
        return f"ReferenceFrame(name={self.name!r}, v={self.v})"
