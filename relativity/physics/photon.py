"""Photon model."""
from __future__ import annotations
from relativity.constants import C, PLANCK
from relativity.utils import smart_array, smart_norm, smart_sqrt, smart_dot, smart_equal, is_symbolic


class Photon:
    def __init__(self, frequency, direction, frame=None, name="photon", c=C, h=PLANCK):
        self.name = name
        self.frequency = frequency
        self.direction = smart_array(direction)
        self.direction = self.direction / smart_norm(self.direction)
        self.frame = frame
        self.c = c
        self.h = h

    @property
    def wavelength(self): return self.c / self.frequency
    @property
    def energy(self): return self.h * self.frequency
    @property
    def momentum_magnitude(self): return self.energy / self.c
    @property
    def momentum(self): return self.momentum_magnitude * self.direction
    @property
    def velocity(self): return self.c * self.direction
    @property
    def four_momentum(self): return smart_array([self.energy / self.c, *self.momentum])
    @property
    def invariant_mass(self): return 0

    def doppler_shift(self, beta):
        return self.frequency * smart_sqrt((1 - beta) / (1 + beta))

    def is_lightlike(self):
        P = self.four_momentum
        E, p = P[0] * self.c, P[1:]
        return smart_equal(E**2 / self.c**2 - smart_dot(p, p), 0)

    def info(self):
        return {"name": self.name, "frequency": self.frequency, "wavelength": self.wavelength,
                "energy": self.energy, "momentum": self.momentum, "four_momentum": self.four_momentum}

    def __repr__(self):
        mode = "symbolic" if is_symbolic(self.frequency) else "numeric"
        return f"Photon(name={self.name!r}, mode={mode})"
