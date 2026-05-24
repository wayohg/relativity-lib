import numpy as np

from relativity.constants import (
    C,
    PLANCK
)


class Photon:
    """
    Relativistic photon.
    """

    def __init__(
        self,
        frequency,
        direction,
        frame=None,
        name="photon",
        c=C,
        h=PLANCK
    ):

        self.name = name

        self.frequency = float(frequency)

        self.direction = smart_array(direction, dtype=float)

        self.frame = frame

        self.c = c

        self.h = h

        self.direction = (
            self.direction /
            np.linalg.norm(self.direction)
        )

    @property
    def wavelength(self):

        return self.c / self.frequency

    @property
    def energy(self):

        return self.h * self.frequency

    @property
    def momentum_magnitude(self):

        return self.energy / self.c

    @property
    def momentum(self):

        return (
            self.momentum_magnitude *
            self.direction
        )

    @property
    def velocity(self):

        return self.c * self.direction

    @property
    def four_momentum(self):

        E_over_c = self.energy / self.c

        return np.array([
            E_over_c,
            *self.momentum
        ])

    def doppler_shift(self, beta):

        """
        Relativistic Doppler shift.

        Positive beta:
        observer moving away.

        Negative beta:
        observer moving toward.
        """

        factor = np.sqrt(
            (1 - beta) /
            (1 + beta)
        )

        return self.frequency * factor

    def info(self):

        return {
            "name": self.name,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "energy": self.energy,
            "momentum": self.momentum,
            "four_momentum": self.four_momentum
        }

    def __repr__(self):

        return (
            f"Photon(name={self.name}, "
            f"frequency={self.frequency:.3e} Hz)"
        )