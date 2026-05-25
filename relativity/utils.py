"""
Hybrid numeric/symbolic utilities for the relativity package.

Rule of thumb
-------------
Use these helpers inside the library instead of forcing dtype=float or
calling np.dot/np.sqrt/np.linalg.norm directly.  They keep the same public
API working with floats, NumPy arrays and SymPy expressions.
"""

from __future__ import annotations

import numbers
import numpy as np
import sympy as sp


# ============================================================
# TYPE DETECTION
# ============================================================

def is_symbolic(x) -> bool:
    """Return True if *x* contains at least one SymPy object."""
    if isinstance(x, sp.Basic):
        return True
    if isinstance(x, sp.MatrixBase):
        return any(isinstance(v, sp.Basic) for v in list(x))
    if isinstance(x, np.ndarray):
        return any(is_symbolic(v) for v in x.flatten())
    if isinstance(x, (list, tuple, set)):
        return any(is_symbolic(v) for v in x)
    return False


def can_compare(x) -> bool:
    """True when numeric inequalities such as x >= 1 are safe."""
    return not is_symbolic(x)


# ============================================================
# ARRAY / MATRIX CREATION
# ============================================================

def smart_array(data, dtype=None):
    """Create a NumPy array, using dtype=object automatically for symbols."""
    if dtype is None:
        dtype = object if is_symbolic(data) else float
    return np.array(data, dtype=dtype)


def smart_matrix(data, dtype=None):
    return smart_array(data, dtype=dtype)


def smart_eye(n: int, symbolic: bool = False):
    return np.eye(n, dtype=object if symbolic else float)


def smart_zeros(shape, symbolic: bool = False):
    return np.zeros(shape, dtype=object if symbolic else float)


# ============================================================
# ELEMENTARY FUNCTIONS
# ============================================================

def smart_sqrt(x):
    return sp.sqrt(x) if is_symbolic(x) else np.sqrt(x)


def smart_exp(x):
    return sp.exp(x) if is_symbolic(x) else np.exp(x)


def smart_sin(x):
    return sp.sin(x) if is_symbolic(x) else np.sin(x)


def smart_cos(x):
    return sp.cos(x) if is_symbolic(x) else np.cos(x)


def smart_tanh(x):
    return sp.tanh(x) if is_symbolic(x) else np.tanh(x)


def smart_atanh(x):
    return sp.atanh(x) if is_symbolic(x) else np.arctanh(x)


def smart_acosh(x):
    return sp.acosh(x) if is_symbolic(x) else np.arccosh(x)


def smart_abs(x):
    return sp.Abs(x) if is_symbolic(x) else abs(x)


# ============================================================
# LINEAR ALGEBRA
# ============================================================

def smart_dot(a, b):
    a = smart_array(a)
    b = smart_array(b)
    if a.shape != b.shape:
        # Let NumPy raise the useful error for numeric arrays.
        if not (is_symbolic(a) or is_symbolic(b)):
            return np.dot(a, b)
    if is_symbolic(a) or is_symbolic(b):
        total = 0
        for x, y in zip(a.flatten(), b.flatten()):
            total += x * y
        return sp.simplify(total)
    return np.dot(a, b)


def smart_norm(v):
    return smart_sqrt(smart_dot(v, v))


def smart_cross(a, b):
    a = smart_array(a)
    b = smart_array(b)
    return smart_array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def smart_outer(a, b):
    a = smart_array(a)
    b = smart_array(b)
    if is_symbolic(a) or is_symbolic(b):
        return np.array([[ai * bj for bj in b] for ai in a], dtype=object)
    return np.outer(a, b)


def smart_matmul(A, B):
    A = smart_array(A)
    B = smart_array(B)
    if is_symbolic(A) or is_symbolic(B):
        result = sp.Matrix(A) * sp.Matrix(B)
        if result.cols == 1:
            return np.array(result, dtype=object).reshape(result.rows)
        return np.array(result, dtype=object)
    return A @ B


def smart_inverse(M):
    return sp.Matrix(M).inv() if is_symbolic(M) else np.linalg.inv(M)


def smart_det(M):
    return sp.Matrix(M).det() if is_symbolic(M) else np.linalg.det(M)


# ============================================================
# RELATIVISTIC HELPERS
# ============================================================

def gamma_factor(v, c):
    beta2 = smart_dot(v, v) / c**2
    if not is_symbolic(beta2) and beta2 >= 1:
        raise ValueError("Velocidad superlumínica no permitida.")
    return simplify(1 / smart_sqrt(1 - beta2))


def beta_factor(v, c):
    return simplify(smart_norm(v) / c)


def rapidity_from_velocity(v, c):
    return smart_atanh(v / c)


def velocity_from_rapidity(u, c):
    return simplify(c * smart_tanh(u))

def normalize_vector(v, name="vector"):
    v = smart_array(v)
    n = smart_norm(v)

    if not is_symbolic(n) and n == 0:
        raise ValueError(f"{name} cannot be the zero vector.")

    return simplify(v / n)


# ============================================================
# SYMBOLIC UTILITIES
# ============================================================

def _vectorize_symbolic(func, x):
    if isinstance(x, np.ndarray):
        return np.vectorize(func, otypes=[object])(x)
    return func(x)


def simplify(x):
    return _vectorize_symbolic(sp.simplify, x) if is_symbolic(x) else x


def expand(x):
    return _vectorize_symbolic(sp.expand, x) if is_symbolic(x) else x


def factor(x):
    return _vectorize_symbolic(sp.factor, x) if is_symbolic(x) else x


def to_numeric(x, dtype=float):
    """Convert a symbolic/numeric array to a numeric NumPy array when possible."""
    if is_symbolic(x):
        x = np.vectorize(lambda y: float(sp.N(y)), otypes=[float])(smart_array(x, dtype=object))
    return np.array(x, dtype=dtype)


# ============================================================
# COMPARISON
# ============================================================

def smart_equal(a, b, tol=1e-10):
    if is_symbolic(a) or is_symbolic(b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a = smart_array(a)
            b = smart_array(b)
            if a.shape != b.shape:
                return False
            return all(sp.simplify(x - y) == 0 for x, y in zip(a.flatten(), b.flatten()))
        return sp.simplify(a - b) == 0
    return np.allclose(a, b, atol=tol, rtol=tol)


def smart_isclose(a, b, tol=1e-10):
    return smart_equal(a, b, tol=tol)


# ============================================================
# PRETTY PRINT
# ============================================================

def pprint(expr):
    if is_symbolic(expr):
        sp.pprint(expr)
    else:
        print(expr)
