import sympy as sp
import numpy as np

from relativity.constants import C

from relativity.utils import (
    smart_array,
    smart_norm,
    smart_sqrt,
    smart_dot,
    gamma_factor,
    beta_factor,
    is_symbolic,
    simplify,
    normalize_vector
)


class Particle:

    """
    Relativistic particle with
    numeric/symbolic hybrid support.
    """

    def __init__(
        self,
        mass,
        position=None,
        velocity=None,
        frame=None,
        name="particle",
        c=C
    ):

        if not is_symbolic(mass) and mass <= 0:
            raise ValueError("Particle mass must be positive. Use Photon for massless particles.")
    
        self.name = name

        self.mass = mass

        self.position = smart_array(
            position if position is not None
            else [0, 0, 0]
        )

        self.velocity = smart_array(
            velocity if velocity is not None
            else [0, 0, 0]
        )

        self.frame = frame

        self.c = c

        self._check_velocity()


    # =====================================================
    # ALTERNATIVE CONSTRUCTORS
    # =====================================================

    @classmethod
    def at_rest(cls, mass, position=None, frame=None, name="particle", c=C):
        return cls(mass=mass, position=position, velocity=[0, 0, 0], frame=frame, name=name, c=c)

    @classmethod
    def from_momentum(cls, mass, momentum, position=None, frame=None, name="particle", c=C):
        momentum = smart_array(momentum)
        p2 = smart_dot(momentum, momentum)
        energy = smart_sqrt(mass**2 * c**4 + p2 * c**2)
        velocity = simplify(momentum * c**2 / energy)
        return cls(mass=mass, position=position, velocity=velocity, frame=frame, name=name, c=c)

    @classmethod
    def from_energy(cls, mass, energy, direction, position=None, frame=None, name="particle", c=C):
        direction = normalize_vector(direction, name="particle direction")
        p_mag = smart_sqrt(energy**2 / c**2 - mass**2 * c**2)
        momentum = p_mag * direction
        return cls.from_momentum(mass, momentum, position=position, frame=frame, name=name, c=c)

    # =====================================================
    # VALIDATION
    # =====================================================

    def _check_velocity(self):

        if is_symbolic(self.velocity) or is_symbolic(self.c):
            return

        speed = self.speed

        if speed >= self.c:

            raise ValueError(
                "Massive particles cannot "
                "reach or exceed c."
            )

    # =====================================================
    # BASIC RELATIVISTIC QUANTITIES
    # =====================================================

    @property
    def speed(self):

        return smart_norm(
            self.velocity
        )

    @property
    def beta(self):

        return beta_factor(
            self.velocity,
            self.c
        )

    @property
    def gamma(self):

        return gamma_factor(
            self.velocity,
            self.c
        )

    # =====================================================
    # ENERGY AND MOMENTUM
    # =====================================================

    @property
    def momentum(self):

        return (
            self.gamma *
            self.mass *
            self.velocity
        )

    @property
    def energy(self):

        return (
            self.gamma *
            self.mass *
            self.c**2
        )

    @property
    def rest_energy(self):

        return (
            self.mass *
            self.c**2
        )

    @property
    def kinetic_energy(self):

        return (
            (self.gamma - 1) *
            self.mass *
            self.c**2
        )

    # =====================================================
    # FOUR MOMENTUM
    # =====================================================

    @property
    def four_momentum(self):

        E_over_c = (
            self.energy / self.c
        )

        return smart_array([

            E_over_c,
            *self.momentum

        ])

    # =====================================================
    # INVARIANT MASS CHECK
    # =====================================================

    @property
    def invariant_mass(self):

        P = self.four_momentum

        E = P[0] * self.c

        p = P[1:]

        m2 = (

            E**2 / self.c**4

            -

            smart_dot(p, p) / self.c**2

        )

        return simplify(
            smart_sqrt(m2)
        )

    # =====================================================
    # SPACETIME HELPERS
    # =====================================================

    def rapidity(self):

        beta = self.beta

        if is_symbolic(beta):

            return sp.atanh(beta)

        return np.arctanh(beta)

    # =====================================================
    # INFO
    # =====================================================

    def info(self):

        return {

            "name": self.name,

            "mass": self.mass,

            "speed": self.speed,

            "beta": self.beta,

            "gamma": self.gamma,

            "energy": self.energy,

            "kinetic_energy":
                self.kinetic_energy,

            "momentum":
                self.momentum,

            "four_momentum":
                self.four_momentum

        }

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):

        mode = (
            "symbolic"
            if is_symbolic(self.velocity)
            else "numeric"
        )

        return (

            f"Particle("
            f"name={self.name}, "
            f"mode={mode})"

        )