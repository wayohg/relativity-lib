import numpy as np
import sympy as sp


# ============================================================
# TYPE DETECTION
# ============================================================

def is_symbolic(x):
    """
    Detect if x contains symbolic SymPy objects.
    """

    if isinstance(x, sp.Basic):
        return True

    if isinstance(x, np.ndarray):

        return any(
            isinstance(v, sp.Basic)
            for v in x.flatten()
        )

    if isinstance(x, (list, tuple)):

        return any(
            is_symbolic(v)
            for v in x
        )

    return False


# ============================================================
# ARRAY CREATION
# ============================================================

def smart_array(data):
    """
    Create symbolic or numeric array automatically.
    """

    dtype = object if is_symbolic(data) else float

    return np.array(data, dtype=dtype)


# ============================================================
# MATHEMATICAL FUNCTIONS
# ============================================================

def smart_sqrt(x):

    if is_symbolic(x):
        return sp.sqrt(x)

    return np.sqrt(x)


def smart_exp(x):

    if is_symbolic(x):
        return sp.exp(x)

    return np.exp(x)


def smart_sin(x):

    if is_symbolic(x):
        return sp.sin(x)

    return np.sin(x)


def smart_cos(x):

    if is_symbolic(x):
        return sp.cos(x)

    return np.cos(x)


def smart_tanh(x):

    if is_symbolic(x):
        return sp.tanh(x)

    return np.tanh(x)


def smart_acosh(x):

    if is_symbolic(x):
        return sp.acosh(x)

    return np.arccosh(x)


# ============================================================
# LINEAR ALGEBRA
# ============================================================

def smart_dot(a, b):

    a = smart_array(a)
    b = smart_array(b)

    if is_symbolic(a) or is_symbolic(b):

        total = 0

        for x, y in zip(a, b):
            total += x * y

        return sp.simplify(total)

    return np.dot(a, b)


def smart_norm(v):

    return smart_sqrt(
        smart_dot(v, v)
    )


def smart_cross(a, b):

    a = smart_array(a)
    b = smart_array(b)

    return smart_array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])


# ============================================================
# RELATIVISTIC HELPERS
# ============================================================

def gamma_factor(v, c):

    beta2 = smart_dot(v, v) / c**2

    return 1 / smart_sqrt(
        1 - beta2
    )


def beta_factor(v, c):

    return smart_norm(v) / c


def rapidity_from_velocity(v, c):

    beta = v / c

    if is_symbolic(beta):

        return sp.atanh(beta)

    return np.arctanh(beta)


def velocity_from_rapidity(u, c):

    return c * smart_tanh(u)


# ============================================================
# SYMBOLIC UTILITIES
# ============================================================

def simplify(x):

    if is_symbolic(x):
        return sp.simplify(x)

    return x


def expand(x):

    if is_symbolic(x):
        return sp.expand(x)

    return x


def factor(x):

    if is_symbolic(x):
        return sp.factor(x)

    return x


# ============================================================
# COMPARISON
# ============================================================

def smart_equal(a, b, tol=1e-10):

    if is_symbolic(a) or is_symbolic(b):

        return sp.simplify(a - b) == 0

    return np.allclose(a, b, atol=tol)


# ============================================================
# MATRIX UTILITIES
# ============================================================

def smart_inverse(M):

    if is_symbolic(M):

        return sp.Matrix(M).inv()

    return np.linalg.inv(M)


def smart_det(M):

    if is_symbolic(M):

        return sp.Matrix(M).det()

    return np.linalg.det(M)


# ============================================================
# PRETTY PRINT
# ============================================================

def pprint(expr):

    if is_symbolic(expr):

        sp.pprint(expr)

    else:

        print(expr)