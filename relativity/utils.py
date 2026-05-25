"""Small numeric/symbolic helpers used across the package."""
from __future__ import annotations
import numpy as np
import sympy as sp


def is_symbolic(x) -> bool:
    if isinstance(x, sp.Basic):
        return True
    if isinstance(x, np.ndarray):
        return any(is_symbolic(v) for v in x.flat)
    if isinstance(x, (list, tuple)):
        return any(is_symbolic(v) for v in x)
    return False


def smart_array(data, dtype=None):
    if dtype is None:
        dtype = object if is_symbolic(data) else float
    return np.array(data, dtype=dtype)


def _symbolic_or_np(x, sym_fn, np_fn):
    return sym_fn(x) if is_symbolic(x) else np_fn(x)


def smart_sqrt(x): return _symbolic_or_np(x, sp.sqrt, np.sqrt)
def smart_exp(x): return _symbolic_or_np(x, sp.exp, np.exp)
def smart_sin(x): return _symbolic_or_np(x, sp.sin, np.sin)
def smart_cos(x): return _symbolic_or_np(x, sp.cos, np.cos)
def smart_tanh(x): return _symbolic_or_np(x, sp.tanh, np.tanh)
def smart_acosh(x): return _symbolic_or_np(x, sp.acosh, np.arccosh)


def simplify(x): return sp.simplify(x) if is_symbolic(x) else x
def expand(x): return sp.expand(x) if is_symbolic(x) else x
def factor(x): return sp.factor(x) if is_symbolic(x) else x


def smart_dot(a, b):
    a = smart_array(a)
    b = smart_array(b)
    if is_symbolic(a) or is_symbolic(b):
        return sp.simplify(sum(ai * bi for ai, bi in zip(a, b)))
    return float(np.dot(a, b))


def smart_norm(v):
    return smart_sqrt(smart_dot(v, v))


def smart_cross(a, b):
    a, b = smart_array(a), smart_array(b)
    return smart_array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def smart_equal(a, b, tol=1e-10):
    if is_symbolic(a) or is_symbolic(b):
        return sp.simplify(a - b) == 0
    return bool(np.allclose(a, b, atol=tol, rtol=tol))


def smart_inverse(M):
    return sp.Matrix(M).inv() if is_symbolic(M) else np.linalg.inv(np.array(M, dtype=float))


def smart_det(M):
    return sp.Matrix(M).det() if is_symbolic(M) else float(np.linalg.det(np.array(M, dtype=float)))


def gamma_factor(v, c):
    beta2 = smart_dot(v, v) / c**2
    if not is_symbolic(beta2) and beta2 >= 1:
        raise ValueError("La rapidez debe ser menor que c.")
    return 1 / smart_sqrt(1 - beta2)


def beta_factor(v, c):
    return smart_norm(v) / c


def rapidity_from_velocity(v, c):
    beta = v / c
    return sp.atanh(beta) if is_symbolic(beta) else np.arctanh(beta)


def velocity_from_rapidity(phi, c):
    return c * smart_tanh(phi)


def pprint(expr):
    sp.pprint(expr) if is_symbolic(expr) else print(expr)
