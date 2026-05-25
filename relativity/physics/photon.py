import sympy as sp

from relativity.constants import (
    C,
    PLANCK
)

from relativity.utils import (
    smart_array,
    smart_norm,
    smart_sqrt,
    smart_dot,
    smart_equal,
    is_symbolic,
    simplify
)


class Photon:

    """
    Relativistic photon with
    symbolic/numeric support.
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

        self.frequency = frequency

        self.direction = smart_array(
            direction
        )

        self.frame = frame

        self.c = c

        self.h = h

        self.direction = (
            self.direction /
            smart_norm(self.direction)
        )

    # =====================================================
    # BASIC QUANTITIES
    # =====================================================

    @property
    def wavelength(self):

        return (
            self.c /
            self.frequency
        )

    @property
    def energy(self):

        return (
            self.h *
            self.frequency
        )

    @property
    def momentum_magnitude(self):

        return (
            self.energy /
            self.c
        )

    @property
    def momentum(self):

        return (

            self.momentum_magnitude *
            self.direction

        )

    @property
    def velocity(self):

        return (
            self.c *
            self.direction
        )

    # =====================================================
    # FOUR MOMENTUM
    # =====================================================

    @property
    def four_momentum(self):

        E_over_c = (
            self.energy /
            self.c
        )

        return smart_array([

            E_over_c,
            *self.momentum

        ])

    # =====================================================
    # NULL INTERVAL CHECK
    # =====================================================

    @property
    def invariant_mass(self):

        return 0

    # =====================================================
    # DOPPLER SHIFT
    # =====================================================

    def doppler_shift(self, beta):

        factor = smart_sqrt(

            (1 - beta)

            /

            (1 + beta)

        )

        return (
            self.frequency *
            factor
        )

    # =====================================================
    # CHECK LIGHTLIKE
    # =====================================================

    def is_lightlike(self):

        P = self.four_momentum

        E = P[0] * self.c

        p = P[1:]

        invariant = (

            E**2 / self.c**2

            -

            smart_dot(p, p)

        )

        return smart_equal(
            invariant,
            0
        )

    # =====================================================
    # INFO
    # =====================================================

    def info(self):

        return {

            "name": self.name,

            "frequency":
                self.frequency,

            "wavelength":
                self.wavelength,

            "energy":
                self.energy,

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
            if is_symbolic(self.frequency)
            else "numeric"
        )

        return (

            f"Photon("
            f"name={self.name}, "
            f"mode={mode})"

        )