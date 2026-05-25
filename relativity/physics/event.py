"""Spacetime event."""

from __future__ import annotations

from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import eta
from relativity.utils import smart_sqrt, smart_array, smart_dot, smart_matmul, smart_equal, is_symbolic, simplify


class Event:
    def __init__(self, t, r, frame=None, c=1.0):
        self.frame = frame
        self.c = c
        r = smart_array(r)
        self.fourvec = FourVector.from_time_space(t, r, c)

    @property
    def t(self):
        return simplify(self.fourvec.ct / self.c)

    @property
    def position(self):
        return self.fourvec.spatial

    @property
    def r(self):
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

    def in_frame(self, new_frame):
        if self.frame is None:
            raise ValueError("Este evento no tiene frame inicial; no se puede transformar automáticamente.")
        new_fourvec = self.frame._transform(self.fourvec, new_frame)
        return Event(new_fourvec.ct / self.c, new_fourvec.spatial, new_frame, self.c)

    def transform(self, frame):
        return self.in_frame(frame)

    def interval_squared(self):
        return self.fourvec.interval_squared()

    def _delta_s2_to(self, other):
        delta = other.fourvec.vec - self.fourvec.vec
        return simplify(smart_dot(delta, smart_matmul(eta, delta)))

    def proper_length_to(self, other):
        s2 = self._delta_s2_to(other)
        if not is_symbolic(s2) and s2 >= 0:
            raise ValueError("Intervalo no es space-like.")
        return simplify(smart_sqrt(-s2) / self.c)

    def proper_time_to(self, other):
        s2 = self._delta_s2_to(other)
        if not is_symbolic(s2) and s2 <= 0:
            raise ValueError("Intervalo tipo espacio o nulo: no hay tiempo propio real.")
        return simplify(smart_sqrt(s2) / self.c)

    def simultaneous_in_frame(self, other, frame):
        delta = other.fourvec.vec - self.fourvec.vec
        u = frame.four_velocity()
        return smart_equal(smart_dot(u.vec, delta), 0)

    def __repr__(self):
        return f"Event(t={self.t}, x={self.x}, y={self.y}, z={self.z})"
