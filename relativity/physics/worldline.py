import numpy as np

from relativity.constants import C


class Worldline:

    def __init__(self, events=None, c=C):

        self.events = events if events is not None else []

        self.c = c

    def add_event(self, event):

        self.events.append(event)

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

        return self.c**2 * dt**2 - np.dot(dr, dr)

    def proper_time(self):

        """
        Total proper time along worldline.
        """

        if len(self.events) < 2:
            return 0.0

        tau = 0.0

        for i in range(len(self.events) - 1):

            e1 = self.events[i]
            e2 = self.events[i + 1]

            interval = self.spacetime_interval(e1, e2)

            if interval > 0:

                tau += smart_sqrt(interval) / self.c

        return tau

    def velocities(self):

        """
        Compute coordinate velocities between events.
        """

        vels = []

        for i in range(len(self.events) - 1):

            e1 = self.events[i]
            e2 = self.events[i + 1]

            dt = e2.t - e1.t

            dr = e2.r - e1.r

            vels.append(dr / dt)

        return vels

    def gamma_factors(self):

        gammas = []

        for v in self.velocities():

            speed = np.linalg.norm(v)

            beta = speed / self.c

            gamma = 1 / smart_sqrt(1 - beta**2)

            gammas.append(gamma)

        return gammas

    def boost(self, frame):

        boosted = []

        for event in self.events:

            boosted.append(event.transform(frame))

        return Worldline(boosted)

    def __repr__(self):

        return (
            f"Worldline(num_events={len(self.events)})"
        )