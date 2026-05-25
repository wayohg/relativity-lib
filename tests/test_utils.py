"""Tests for the hybrid numeric/symbolic utility layer."""
from __future__ import annotations

import numpy as np
import sympy as sp

from relativity.utils import (
    is_symbolic,
    smart_array,
    smart_dot,
    smart_norm,
    smart_sqrt,
    smart_matmul,
    simplify,
    smart_equal,
)


def test_smart_array_numeric_uses_float_dtype():
    arr = smart_array([1, 2, 3])
    assert arr.dtype == float
    assert np.allclose(arr, [1.0, 2.0, 3.0])


def test_smart_array_symbolic_uses_object_dtype():
    x = sp.Symbol("x")
    arr = smart_array([x, 1, 2])
    assert arr.dtype == object
    assert is_symbolic(arr)


def test_smart_dot_numeric():
    assert smart_dot([1, 2, 3], [4, 5, 6]) == 32


def test_smart_dot_symbolic():
    x, y = sp.symbols("x y")
    result = smart_dot([x, y, 0], [x, -y, 0])
    assert sp.simplify(result - (x**2 - y**2)) == 0


def test_smart_norm_symbolic():
    x = sp.Symbol("x", positive=True)
    assert smart_equal(smart_norm([x, 0, 0]), x)


def test_smart_sqrt_symbolic():
    x = sp.Symbol("x")
    assert smart_sqrt(x**2) == sp.sqrt(x**2)


def test_smart_matmul_symbolic_matrix_vector():
    x = sp.Symbol("x")
    A = [[1, x], [0, 1]]
    b = [2, 3]
    result = smart_matmul(A, b)
    assert result.shape == (2,)
    assert sp.simplify(result[0] - (2 + 3*x)) == 0
    assert sp.simplify(result[1] - 3) == 0


def test_simplify_vectorized_symbolic_array():
    x = sp.Symbol("x")
    arr = smart_array([sp.sin(x)**2 + sp.cos(x)**2, x + x])
    out = simplify(arr)
    assert out[0] == 1
    assert out[1] == 2*x
