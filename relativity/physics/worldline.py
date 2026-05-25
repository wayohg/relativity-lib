"""Worldline represented as an ordered sequence of events."""

from __future__ import annotations

import numpy as np

from relativity.constants import C
from relativity.utils import smart_array, smart_dot, smart_sqrt, gamma_factor, is_symbolic, simplify


class Worldline:
    def __init__(self, events=None, c=C):
        self.events = []
        self.c = c

        for event in events or []:
            self.add_event(event)

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
        """
        Compute the accumulated proper time along the worldline.
    
        Uses the convention:
    
            ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2
    
        Therefore:
    
            dτ = sqrt(ds^2) / c
    
        Raises
        ------
        ValueError
            If any numeric segment is spacelike.
        """
        from relativity.utils import (
            smart_array,
            smart_dot,
            smart_sqrt,
            is_symbolic,
            simplify,
        )
    
        tau = 0
    
        for i in range(len(self.events) - 1):
            e1 = self.events[i]
            e2 = self.events[i + 1]
    
            dt = e2.t - e1.t
    
            r1 = smart_array([e1.x, e1.y, e1.z])
            r2 = smart_array([e2.x, e2.y, e2.z])
    
            dr = r2 - r1
    
            interval = self.c**2 * dt**2 - smart_dot(dr, dr)
    
            if is_symbolic(interval):
                tau += smart_sqrt(interval) / self.c
            elif interval > 0:
                tau += smart_sqrt(interval) / self.c
            elif interval == 0:
                continue
            else:
                raise ValueError(
                    f"Segment {i}->{i + 1} is spacelike; proper time is not real."
                )
    
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
