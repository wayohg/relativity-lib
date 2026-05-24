import numpy as np


class Tensor:

    """
    Generic tensor object.
    """

    def __init__(self, components):

        self.components = np.array(
            components,
            dtype=float
        )

    @property
    def shape(self):

        return self.components.shape

    @property
    def rank(self):

        return len(self.shape)

    def __add__(self, other):

        return Tensor(
            self.components + other.components
        )

    def __sub__(self, other):

        return Tensor(
            self.components - other.components
        )

    def __mul__(self, scalar):

        return Tensor(
            self.components * scalar
        )

    def tensor_product(self, other):

        return Tensor(
            np.tensordot(
                self.components,
                other.components,
                axes=0
            )
        )

    def contract(self, axis1, axis2):

        return Tensor(
            np.trace(
                self.components,
                axis1=axis1,
                axis2=axis2
            )
        )

    def transpose(self, axes=None):

        return Tensor(
            np.transpose(
                self.components,
                axes=axes
            )
        )

    def __repr__(self):

        return (
            f"Tensor(rank={self.rank}, "
            f"shape={self.shape})"
        )
    
class MinkowskiMetric(Tensor):

    def __init__(self, signature="+---"):

        if signature == "+---":

            eta = np.diag([1, -1, -1, -1])

        elif signature == "-+++":

            eta = np.diag([-1, 1, 1, 1])

        else:

            raise ValueError(
                "Invalid signature."
            )

        super().__init__(eta)

        self.signature = signature

class ElectromagneticTensor(Tensor):

    def __init__(self, E, B):

        Ex, Ey, Ez = E
        Bx, By, Bz = B

        F = np.array([

            [0,   Ex,   Ey,   Ez],
            [-Ex, 0,    Bz,  -By],
            [-Ey,-Bz,   0,    Bx],
            [-Ez, By,  -Bx,   0]

        ])

        super().__init__(F)

        self.E = smart_array(E)
        self.B = smart_array(B)