import numpy as np

from relativity.constants import C
from relativity.utils import smart_sqrt


class Worldline:
    """
    Sequence of spacetime events representing a particle's history.
    Events are kept sorted by coordinate time.
    """

    def __init__(self, events=None, c=C):
        self.events = events if events is not None else []
        self.c = c

    # =====================================================
    # EVENT MANAGEMENT
    # =====================================================

    def add_event(self, event):
        self.events.append(event)
        self.events.sort(key=lambda e: e.t)

    # =====================================================
    # COORDINATE ARRAYS
    # =====================================================

    @property
    def times(self):
        return np.array([e.t for e in self.events])

    @property
    def positions(self):
        return np.array([e.r for e in self.events])  # FIX: requires Event.r property

    # =====================================================
    # INTERVALS AND PROPER TIME
    # =====================================================

    def spacetime_interval(self, e1, e2):
        """Coordinate-frame interval s² = c²Δt² - |Δr|²."""
        dt = e2.t - e1.t
        dr = e2.r - e1.r
        return self.c**2 * dt**2 - np.dot(dr, dr)

    def proper_time(self):
        """Total proper time τ along the worldline."""
        if len(self.events) < 2:
            return 0.0

        tau = 0.0
        for i in range(len(self.events) - 1):
            e1, e2 = self.events[i], self.events[i + 1]
            interval = self.spacetime_interval(e1, e2)
            if interval > 0:
                tau += smart_sqrt(interval) / self.c  # FIX: use smart_sqrt for consistency
        return tau

    # =====================================================
    # KINEMATICS
    # =====================================================

    def velocities(self):
        """Coordinate velocities between consecutive events."""
        vels = []
        for i in range(len(self.events) - 1):
            e1, e2 = self.events[i], self.events[i + 1]
            dt = e2.t - e1.t
            if dt == 0:
                raise ValueError(      # FIX: was silently returning dr/0
                    f"Events {i} and {i+1} have identical coordinate time."
                )
            dr = e2.r - e1.r
            vels.append(dr / dt)
        return vels

    def gamma_factors(self):
        """Lorentz γ factors between consecutive events."""
        gammas = []
        for v in self.velocities():
            speed = np.linalg.norm(v)
            beta  = speed / self.c
            if beta >= 1.0:
                raise ValueError(
                    "Superluminal segment detected in worldline."
                )
            gammas.append(1.0 / np.sqrt(1 - beta**2))
        return gammas

    # =====================================================
    # BOOST
    # =====================================================

    def boost(self, frame):
        """Return a new Worldline with all events transformed into frame."""
        boosted = [event.transform(frame) for event in self.events]  # FIX: was event.transform() undefined
        return Worldline(boosted, c=self.c)

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        return f"Worldline(num_events={len(self.events)})"
