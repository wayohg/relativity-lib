import numpy as np

from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import eta            # FIX: was importing ETA (wrong name)
from relativity.utils import smart_sqrt, smart_array


class Event:
    def __init__(self, t, r, frame=None, c=1.0):   # FIX: frame/c now have defaults
        self.frame = frame
        self.c = float(c)
        r = smart_array(r, dtype=float)
        self.fourvec = FourVector.from_time_space(t, r, c)

    # =====================================================
    # COORDINATE PROPERTIES
    # =====================================================

    @property
    def t(self):
        return self.fourvec.ct / self.c

    @property
    def position(self):
        return self.fourvec.spatial

    @property
    def r(self):                                     # FIX: worldline.py accesses e.r
        return self.fourvec.spatial

    @property
    def x(self):
        return self.fourvec.vec[1]

    @property
    def y(self):
        return self.fourvec.vec[2]

    @property
    def z(self):
        return self.fourvec.vec[3]

    # =====================================================
    # FRAME TRANSFORMATION
    # =====================================================

    def in_frame(self, new_frame):
        """Transform this event into another reference frame."""
        new_fourvec = self.frame._transform(self.fourvec, new_frame)
        t_new = new_fourvec.ct / self.c
        return Event(t_new, new_fourvec.spatial, new_frame, self.c)

    def transform(self, frame):                      # FIX: worldline.boost() calls event.transform()
        """Alias for in_frame; used by Worldline.boost."""
        return self.in_frame(frame)

    # =====================================================
    # INVARIANT QUANTITIES
    # =====================================================

    def interval_squared(self):
        return self.fourvec.interval_squared()

    def proper_length_to(self, other):
        """
        Longitud propia entre dos eventos.
        Solo válida si el intervalo es space-like (s² < 0).
        """
        delta = other.fourvec.vec - self.fourvec.vec
        s2 = delta @ eta @ delta                     # FIX: ETA → eta

        if s2 >= 0:
            raise ValueError("Intervalo no es space-like.")

        return smart_sqrt(-s2) / self.c

    def proper_time_to(self, other):
        """
        Tiempo propio entre dos eventos (invariante).
        Solo válido si el intervalo es time-like (s² > 0).
        """
        delta = other.fourvec.vec - self.fourvec.vec
        s2 = delta @ eta @ delta                     # FIX: ETA → eta

        if s2 <= 0:
            raise ValueError("Intervalo tipo espacio o nulo: no hay tiempo propio real.")

        return smart_sqrt(s2) / self.c

    def simultaneous_in_frame(self, other, frame):
        """True if these two events are simultaneous in the given frame."""
        delta = other.fourvec.vec - self.fourvec.vec
        u = frame.four_velocity()
        return np.isclose(np.dot(u.vec, delta), 0)  # FIX: u is FourVector, need u.vec

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        return (
            f"Event(t={self.t:.4g}, "
            f"x={self.x:.4g}, y={self.y:.4g}, z={self.z:.4g})"
        )
