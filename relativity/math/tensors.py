"""Basic tensor containers and the Minkowski metric."""
from __future__ import annotations
import numpy as np
import sympy as sp
from relativity.utils import smart_array, is_symbolic


class Tensor:
    def __init__(self, components):
        self.components = smart_array(components)

    @property
    def shape(self): return self.components.shape

    @property
    def rank(self): return len(self.shape)

    @property
    def symbolic(self): return is_symbolic(self.components)

    def __getitem__(self, key): return self.components[key]
    def __array__(self, dtype=None): return np.asarray(self.components, dtype=dtype)

    def __add__(self, other): return Tensor(self.components + other.components)
    def __sub__(self, other): return Tensor(self.components - other.components)
    def __mul__(self, scalar): return Tensor(self.components * scalar)
    __rmul__ = __mul__

    def tensor_product(self, other):
        if self.symbolic or other.symbolic:
            return Tensor(sp.tensorproduct(sp.Array(self.components), sp.Array(other.components)))
        return Tensor(np.tensordot(self.components, other.components, axes=0))

    def contract(self, axis1, axis2):
        if self.symbolic:
            return Tensor(sp.tensorcontraction(sp.Array(self.components), (axis1, axis2)))
        return Tensor(np.trace(self.components, axis1=axis1, axis2=axis2))

    def transpose(self, axes=None):
        if self.symbolic:
            return Tensor(sp.permutedims(sp.Array(self.components), axes))
        return Tensor(np.transpose(self.components, axes=axes))

    def __repr__(self):
        kind = "symbolic" if self.symbolic else "numeric"
        return f"Tensor(rank={self.rank}, shape={self.shape}, type={kind})"


class MinkowskiMetric(Tensor):
    """Metric with signature (+---). Coordinates are x^mu=(ct,x,y,z)."""
    def __init__(self):
        super().__init__(np.diag([1, -1, -1, -1]))

    def lower(self, v): return self.components @ smart_array(v)
    def raise_(self, v): return self.components @ smart_array(v)


class ElectromagneticTensor(Tensor):
    """Contravariant F^{mu nu} using SI-like convention with E/c entries."""
    def __init__(self, E, B, c=1.0):
        Ex, Ey, Ez = smart_array(E)
        Bx, By, Bz = smart_array(B)
        super().__init__([
            [0, -Ex/c, -Ey/c, -Ez/c],
            [Ex/c, 0, -Bz, By],
            [Ey/c, Bz, 0, -Bx],
            [Ez/c, -By, Bx, 0],
        ])
