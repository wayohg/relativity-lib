"""Massive relativistic particle."""
from __future__ import annotations
import numpy as np
import sympy as sp
from relativity.constants import C
from relativity.utils import smart_array, smart_norm, smart_sqrt, smart_dot, gamma_factor, beta_factor, is_symbolic, simplify


class Particle:
    def __init__(self, mass, position=None, velocity=None, frame=None, name="particle", c=C):
        self.name = name
        self.mass = mass
        self.position = smart_array(position if position is not None else [0, 0, 0])
        self.velocity = smart_array(velocity if velocity is not None else [0, 0, 0])
        self.frame = frame
        self.c = c
        self._check_velocity()

    def _check_velocity(self):
        if is_symbolic(self.velocity):
            return
        if self.speed >= self.c:
            raise ValueError("Una partícula masiva debe tener |v| < c.")

    @property
    def speed(self): return smart_norm(self.velocity)
    @property
    def beta(self): return beta_factor(self.velocity, self.c)
    @property
    def gamma(self): return gamma_factor(self.velocity, self.c)
    @property
    def momentum(self): return self.gamma * self.mass * self.velocity
    @property
    def energy(self): return self.gamma * self.mass * self.c**2
    @property
    def rest_energy(self): return self.mass * self.c**2
    @property
    def kinetic_energy(self): return (self.gamma - 1) * self.rest_energy
    @property
    def four_momentum(self): return smart_array([self.energy / self.c, *self.momentum])

    @property
    def invariant_mass(self):
        P = self.four_momentum
        E, p = P[0] * self.c, P[1:]
        return simplify(smart_sqrt(E**2 / self.c**4 - smart_dot(p, p) / self.c**2))

    def rapidity(self):
        return sp.atanh(self.beta) if is_symbolic(self.beta) else np.arctanh(self.beta)

    def info(self):
        return {"name": self.name, "mass": self.mass, "speed": self.speed, "beta": self.beta,
                "gamma": self.gamma, "energy": self.energy, "kinetic_energy": self.kinetic_energy,
                "momentum": self.momentum, "four_momentum": self.four_momentum}

    def __repr__(self):
        mode = "symbolic" if is_symbolic(self.velocity) else "numeric"
        return f"Particle(name={self.name!r}, mode={mode})"
