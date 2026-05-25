import numpy as np

from relativity.physics.fourvector import FourVector
from relativity.math.lorentz import boost_matrix
from relativity.math import relativistic_velocity_addition
from relativity.utils import smart_array, smart_sqrt


class ReferenceFrame:
    def __init__(self, name, velocity, relative_to=None, c=1.0):
        self.name = name
        self.v = smart_array(velocity, dtype=float)  # FIX: smart_array was undefined (missing import)
        self.c = float(c)
        self.relative_to = relative_to

    # =====================================================
    # VELOCITY PROPERTIES
    # =====================================================

    @property
    def velocity(self):
        return self.v

    @property
    def beta(self):
        return self.v / self.c

    # =====================================================
    # VELOCITY COMPOSITION
    # =====================================================

    def _velocity_wrt_root(self):
        """Absolute velocity relative to the root (inertial) frame."""
        if self.relative_to is None:
            return self.v
        parent_v = self.relative_to._velocity_wrt_root()
        return relativistic_velocity_addition(self.v, parent_v, self.c)

    def velocity_wrt(self, other_frame):
        """
        Velocity of other_frame as seen from self.
        Uses relativistic velocity subtraction via addition with negated velocity.
        """
        v_self  = self._velocity_wrt_root()
        v_other = other_frame._velocity_wrt_root()
        # FIX: correct sign convention — we want "other relative to self"
        return relativistic_velocity_addition(v_other, -v_self, self.c)

    # =====================================================
    # LORENTZ TRANSFORMATION
    # =====================================================

    def _transformation_to(self, other):
        """Boost matrix that maps coordinates from self → other."""
        v_rel = self.velocity_wrt(other)
        return boost_matrix(v_rel, self.c)

    def _transform(self, fourvector, other):
        """Apply boost to a FourVector, returning a new FourVector."""
        Lambda = self._transformation_to(other)
        new_vec = Lambda @ fourvector.vec
        return FourVector(*new_vec)

    # =====================================================
    # FOUR-VELOCITY
    # =====================================================

    def four_velocity(self):
        """Return the four-velocity U^μ = γ(c, v) of this frame."""
        v2 = np.dot(self.v, self.v)
        if v2 == 0:
            return FourVector(self.c, 0.0, 0.0, 0.0)
        g = 1.0 / smart_sqrt(1 - v2 / self.c**2)
        return FourVector(g * self.c, *(g * self.v))

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        return f"ReferenceFrame(name={self.name!r}, v={self.v})"
