"""Worldline represented as an ordered sequence of events."""

from __future__ import annotations

import numpy as np

from relativity.constants import C
from relativity.utils import smart_array, smart_dot, smart_sqrt, gamma_factor, is_symbolic, simplify


class Worldline:
    def __init__(self, events=None, c=C):
        self.events = events if events is not None else []
        self.c = c

    def add_event(self, event):
        self.events.append(event)
        # Sort only when the times are numeric. Symbolic ordering is undefined.
        if all(not is_symbolic(e.t) for e in self.events):
            self.events.sort(key=lambda e: e.t)

    @property
    def times(self):
        return smart_array([e.t for e in self.events])

    @property
    def positions(self):
        return smart_array([e.r for e in self.events])

    def spacetime_interval(self, e1, e2):
        dt = e2.t - e1.t
        dr = e2.r - e1.r
        return simplify(self.c**2 * dt**2 - smart_dot(dr, dr))

    def proper_time(self):
        if len(self.events) < 2:
            return 0
        tau = 0
        for i in range(len(self.events) - 1):
            interval = self.spacetime_interval(self.events[i], self.events[i + 1])
            if is_symbolic(interval) or interval > 0:
                tau += smart_sqrt(interval) / self.c
        return simplify(tau)

    def velocities(self):
        vels = []
        for i in range(len(self.events) - 1):
            e1, e2 = self.events[i], self.events[i + 1]
            dt = e2.t - e1.t
            if not is_symbolic(dt) and dt == 0:
                raise ValueError(f"Events {i} and {i + 1} have identical coordinate time.")
            vels.append(simplify((e2.r - e1.r) / dt))
        return vels

    def gamma_factors(self):
        return [gamma_factor(v, self.c) for v in self.velocities()]

    def boost(self, frame):
        return Worldline([event.transform(frame) for event in self.events], c=self.c)

    def __repr__(self):
        return f"Worldline(num_events={len(self.events)})"
