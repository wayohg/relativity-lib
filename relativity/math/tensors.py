import numpy as np
import sympy as sp

from relativity.utils import (
    smart_array,
    is_symbolic,
    smart_dot,
    simplify,
    smart_equal
)


class Tensor:

    """
    Generic tensor object with
    numeric/symbolic hybrid support.
    """

    def __init__(self, components):

        self.components = smart_array(
            components
        )

    # =====================================================
    # BASIC PROPERTIES
    # =====================================================

    @property
    def shape(self):

        return self.components.shape

    @property
    def rank(self):

        return len(self.shape)

    @property
    def symbolic(self):

        return is_symbolic(
            self.components
        )

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):

        tensor_type = (
            "symbolic"
            if self.symbolic
            else "numeric"
        )

        return (
            f"Tensor("
            f"rank={self.rank}, "
            f"shape={self.shape}, "
            f"type={tensor_type})"
        )

    # =====================================================
    # INDEXING
    # =====================================================

    def __getitem__(self, key):

        return self.components[key]

    def __setitem__(self, key, value):

        self.components[key] = value

    # =====================================================
    # ALGEBRA
    # =====================================================

    def __add__(self, other):

        return Tensor(
            self.components +
            other.components
        )

    def __sub__(self, other):

        return Tensor(
            self.components -
            other.components
        )

    def __mul__(self, scalar):

        return Tensor(
            self.components * scalar
        )

    def __rmul__(self, scalar):

        return self.__mul__(scalar)

    # =====================================================
    # TENSOR PRODUCT
    # =====================================================

    def tensor_product(self, other):

        if self.symbolic or other.symbolic:

            A = sp.Array(self.components)
            B = sp.Array(other.components)

            return Tensor(
                sp.tensorproduct(A, B)
            )

        return Tensor(
            np.tensordot(
                self.components,
                other.components,
                axes=0
            )
        )

    # =====================================================
    # CONTRACTION
    # =====================================================

    def contract(self, axis1, axis2):

        if self.symbolic:

            arr = sp.Array(self.components)

            contracted = sp.tensorcontraction(
                arr,
                (axis1, axis2)
            )

            return Tensor(contracted)

        contracted = np.trace(
            self.components,
            axis1=axis1,
            axis2=axis2
        )

        return Tensor(contracted)

    # =====================================================
    # TRANSPOSE
    # =====================================================

    def transpose(self, axes=None):

        if self.symbolic:

            arr = sp.Array(self.components)

            return Tensor(
                sp.permutedims(
                    arr,
                    axes
                )
            )

        return Tensor(
            np.transpose(
                self.components,
                axes=axes
            )
        )

    # =====================================================
    # SIMPLIFICATION
    # =====================================================

    def simplify(self):

        if self.symbolic:

            simplified = np.vectorize(
                sp.simplify
            )(self.components)

            return Tensor(simplified)

        return self

    # =====================================================
    # EQUALITY
    # =====================================================

    def equals(self, other):

        return smart_equal(
            self.components,
            other.components
        )

    # =====================================================
    # MATRIX OPERATIONS
    # =====================================================

    def determinant(self):

        if self.rank != 2:

            raise ValueError(
                "Determinant only defined "
                "for rank-2 tensors."
            )

        if self.symbolic:

            return sp.Matrix(
                self.components
            ).det()

        return np.linalg.det(
            self.components
        )

    def inverse(self):

        if self.rank != 2:

            raise ValueError(
                "Inverse only defined "
                "for rank-2 tensors."
            )

        if self.symbolic:

            inv = sp.Matrix(
                self.components
            ).inv()

            return Tensor(inv)

        return Tensor(
            np.linalg.inv(
                self.components
            )
        )
    
class MinkowskiMetric(Tensor):

    def __init__(self, signature="+---"):

        if signature == "+---":

            eta = [

                [1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0,-1, 0],
                [0, 0, 0,-1]

            ]

        elif signature == "-+++":

            eta = [

                [-1, 0, 0, 0],
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0],
                [ 0, 0, 0, 1]

            ]

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

        F = [

            [0,   Ex,   Ey,   Ez],
            [-Ex, 0,    Bz,  -By],
            [-Ey,-Bz,   0,    Bx],
            [-Ez, By,  -Bx,   0]

        ]

        super().__init__(F)

        self.E = smart_array(E)
        self.B = smart_array(B)