"""Piecewise worldline built from events."""
from __future__ import annotations
import numpy as np
from relativity.constants import C
from relativity.physics.event import Event
from relativity.utils import smart_array, smart_dot, smart_sqrt


class Worldline:
    def __init__(self, events=None, c=C):
        self.events = sorted(events or [], key=lambda e: e.t)
        self.c = c

    @classmethod
    def inertial(cls, position0, velocity, t_values, frame=None, c=C):
        r0, v = smart_array(position0), smart_array(velocity)
        return cls([Event(t, r0 + v * t, frame=frame, c=c) for t in t_values], c=c)

    def add_event(self, event):
        self.events.append(event)
        self.events.sort(key=lambda e: e.t)
        return self

    @property
    def times(self): return smart_array([e.t for e in self.events])
    @property
    def positions(self): return smart_array([e.r for e in self.events])

    def segments(self):
        return zip(self.events[:-1], self.events[1:])

    def proper_time(self):
        tau = 0
        for e1, e2 in self.segments():
            s2 = e1.interval_squared_to(e2)
            if s2 > 0:
                tau += smart_sqrt(s2) / self.c
        return tau

    def velocities(self):
        out = []
        for e1, e2 in self.segments():
            dt = e2.t - e1.t
            if dt == 0:
                raise ValueError("Dos eventos consecutivos tienen el mismo tiempo coordenado.")
            out.append((e2.r - e1.r) / dt)
        return out

    def gamma_factors(self):
        return [1 / smart_sqrt(1 - smart_dot(v, v) / self.c**2) for v in self.velocities()]

    def in_frame(self, new_frame):
        return Worldline([e.in_frame(new_frame) for e in self.events], c=self.c)

    def __repr__(self):
        return f"Worldline(num_events={len(self.events)})"
