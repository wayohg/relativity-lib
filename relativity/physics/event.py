"""Spacetime event."""
from __future__ import annotations
import numpy as np
from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import interval_squared, classify_interval
from relativity.utils import smart_array, smart_sqrt


class Event:
    def __init__(self, t, r, frame=None, c=1.0, name=None):
        self.name = name
        self.frame = frame
        self.c = c
        self.fourvec = FourVector.from_time_space(t, r, c)

    @property
    def t(self): return self.fourvec.ct / self.c
    @property
    def r(self): return self.fourvec.spatial
    @property
    def position(self): return self.r
    @property
    def x(self): return self.r[0]
    @property
    def y(self): return self.r[1]
    @property
    def z(self): return self.r[2]

    def vector_to(self, other):
        return other.fourvec.vec - self.fourvec.vec

    def interval_squared_to(self, other):
        return interval_squared(self.vector_to(other))

    def interval_type_to(self, other):
        return classify_interval(self.vector_to(other))

    def proper_time_to(self, other):
        s2 = self.interval_squared_to(other)
        if s2 < 0:
            raise ValueError("Intervalo tipo espacio: no hay tiempo propio real.")
        return smart_sqrt(s2) / self.c

    def proper_length_to(self, other):
        s2 = self.interval_squared_to(other)
        if s2 > 0:
            raise ValueError("Intervalo tipo tiempo: no hay longitud propia espacial.")
        return smart_sqrt(-s2)

    def in_frame(self, new_frame):
        if self.frame is None:
            raise ValueError("El evento no tiene frame origen.")
        fv = self.frame.transform_fourvector(self.fourvec, new_frame)
        return Event(fv.ct / self.c, fv.spatial, frame=new_frame, c=self.c, name=self.name)

    def __repr__(self):
        fname = self.frame.name if self.frame is not None else None
        return f"Event(t={self.t}, r={self.r}, frame={fname})"
