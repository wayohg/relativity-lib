"""
relativity.quantum.probability
==============================

Small probability/statistics helpers for introductory quantum mechanics.

The functions in this module are useful for normalized continuous
probability densities, discrete distributions, expectation values, moments,
variance, standard deviation, cumulative distributions and simple grid-based
approximations.

Design goals
------------
- Work with both numeric values and SymPy expressions when possible.
- Avoid requiring SciPy.
- Keep a lightweight API compatible with the rest of the project.

Examples
--------
Continuous expectation value:

>>> import sympy as sp
>>> from relativity.quantum.probability import continuous_average
>>> x = sp.symbols("x", real=True)
>>> f = sp.Rational(1, 10) * (10 - x)**2
>>> continuous_average(f, x, 0, 10)
5/2

Discrete expectation value:

>>> from relativity.quantum.probability import discrete_average_from_function
>>> discrete_average_from_function(lambda x: 0.1 * (10 - x)**2, [0, 1, 2, 3])
1.2
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union, Any

import math
import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover - SymPy is expected in this project
    sp = None


NumberLike = Union[int, float, complex]


# -----------------------------------------------------------------------------
# Compatibility helpers
# -----------------------------------------------------------------------------

try:
    from relativity.utils import is_symbolic, simplify
except Exception:
    def is_symbolic(x: Any) -> bool:
        """Fallback symbolic detector."""
        if sp is None:
            return False
        try:
            if isinstance(x, sp.Basic):
                return True
            if isinstance(x, (list, tuple, np.ndarray)):
                return any(is_symbolic(item) for item in np.asarray(x, dtype=object).flat)
        except Exception:
            return False
        return False

    def simplify(x: Any) -> Any:
        """Fallback simplify function."""
        if sp is not None and is_symbolic(x):
            try:
                return sp.simplify(x)
            except Exception:
                return x
        return x


def _require_sympy() -> None:
    if sp is None:
        raise ImportError("SymPy is required for symbolic probability functions.")


def _as_object_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def _as_float_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _is_callable(obj: Any) -> bool:
    return callable(obj)


def _eval_function_or_expr(func_or_expr: Any, variable: Any, values: Any) -> Any:
    """Evaluate either a Python callable or a SymPy expression."""
    if _is_callable(func_or_expr):
        return func_or_expr(values)

    if sp is not None and isinstance(func_or_expr, sp.Basic):
        f = sp.lambdify(variable, func_or_expr, modules="numpy")
        return f(values)

    raise TypeError("Expected a callable or a SymPy expression.")


def _validate_nonnegative_numeric(weights: np.ndarray, name: str = "weights") -> None:
    """Reject negative numeric weights."""
    if any(is_symbolic(w) for w in weights.flat):
        return
    if np.any(np.asarray(weights, dtype=float) < 0):
        raise ValueError(f"{name} must be nonnegative.")


def _safe_sum(values: np.ndarray) -> Any:
    """Sum numeric/object arrays in a symbolic-safe way."""
    total = 0
    for value in np.asarray(values, dtype=object).flat:
        total += value
    return simplify(total)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible with old and new NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return _trapz(y, x)


# -----------------------------------------------------------------------------
# Continuous distributions
# -----------------------------------------------------------------------------

def continuous_integral(
    density,
    variable,
    lower,
    upper,
):
    """
    Integrate a continuous density over an interval.

    Parameters
    ----------
    density : sympy expression
        Function f(x) to integrate.
    variable : sympy.Symbol
        Integration variable.
    lower, upper : numbers or symbolic values
        Bounds of integration.
    """
    _require_sympy()
    result = sp.integrate(density, (variable, lower, upper))
    return simplify(result)


def normalization_constant_continuous(
    density,
    variable,
    lower,
    upper,
):
    """
    Return A such that A * density integrates to 1 over [lower, upper].
    """
    integral = continuous_integral(density, variable, lower, upper)
    if integral == 0:
        raise ValueError("Density integral is zero; cannot normalize.")
    return simplify(1 / integral)


def normalize_continuous_density(
    density,
    variable,
    lower,
    upper,
):
    """
    Return a normalized continuous density over [lower, upper].
    """
    A = normalization_constant_continuous(density, variable, lower, upper)
    return simplify(A * density)


def continuous_expectation(
    observable,
    density,
    variable,
    lower,
    upper,
    normalized: bool = False,
):
    """
    Compute E[g(x)] for a continuous distribution.

    If normalized=False, the function computes

        ∫ g(x) f(x) dx / ∫ f(x) dx.

    If normalized=True, the denominator is assumed to be 1.
    """
    _require_sympy()
    numerator = sp.integrate(observable * density, (variable, lower, upper))

    if normalized:
        return simplify(numerator)

    denominator = sp.integrate(density, (variable, lower, upper))
    if denominator == 0:
        raise ValueError("Density integral is zero; expectation is undefined.")

    return simplify(numerator / denominator)


def continuous_average(
    density,
    variable,
    lower,
    upper,
    normalized: bool = False,
):
    """
    Compute the average value <x> for a continuous density f(x).
    """
    return continuous_expectation(variable, density, variable, lower, upper, normalized=normalized)


def continuous_moment(
    order: int,
    density,
    variable,
    lower,
    upper,
    center=None,
    normalized: bool = False,
):
    """
    Compute a continuous moment.

    If center is None, computes <x^order>.
    If center is given, computes <(x - center)^order>.
    """
    if order < 0:
        raise ValueError("Moment order must be nonnegative.")

    observable = variable**order if center is None else (variable - center)**order
    return continuous_expectation(observable, density, variable, lower, upper, normalized=normalized)


def continuous_variance(
    density,
    variable,
    lower,
    upper,
    normalized: bool = False,
):
    """
    Compute Var(x) = <x²> - <x>² for a continuous density.
    """
    mean = continuous_average(density, variable, lower, upper, normalized=normalized)
    second = continuous_moment(2, density, variable, lower, upper, normalized=normalized)
    return simplify(second - mean**2)


def continuous_std(
    density,
    variable,
    lower,
    upper,
    normalized: bool = False,
):
    """
    Compute the standard deviation for a continuous density.
    """
    _require_sympy()
    return simplify(sp.sqrt(continuous_variance(density, variable, lower, upper, normalized=normalized)))


def continuous_probability(
    density,
    variable,
    lower,
    upper,
    domain_lower=None,
    domain_upper=None,
    normalized: bool = False,
):
    """
    Compute P(lower <= x <= upper) for a continuous density.

    If normalized=False, domain_lower and domain_upper must be provided so the
    function can divide by the total probability in the full domain.
    """
    _require_sympy()
    numerator = sp.integrate(density, (variable, lower, upper))

    if normalized:
        return simplify(numerator)

    if domain_lower is None or domain_upper is None:
        raise ValueError("domain_lower and domain_upper are required when normalized=False.")

    denominator = sp.integrate(density, (variable, domain_lower, domain_upper))
    if denominator == 0:
        raise ValueError("Density integral is zero; probability is undefined.")

    return simplify(numerator / denominator)


def continuous_cdf(
    density,
    variable,
    lower,
    x_value=None,
    upper=None,
    normalized: bool = False,
):
    """
    Compute or return the cumulative distribution F(x).

    If x_value is None, returns a symbolic expression F(x). In that case,
    `upper` is ignored unless normalization is needed.

    If normalized=False, `upper` must be provided to normalize over
    [lower, upper].
    """
    _require_sympy()
    dummy = sp.Symbol("_xi", real=True)
    expr = density.subs(variable, dummy)

    x_symbol = variable if x_value is None else x_value
    numerator = sp.integrate(expr, (dummy, lower, x_symbol))

    if normalized:
        return simplify(numerator)

    if upper is None:
        raise ValueError("upper is required when normalized=False.")

    denominator = sp.integrate(density, (variable, lower, upper))
    if denominator == 0:
        raise ValueError("Density integral is zero; CDF is undefined.")

    return simplify(numerator / denominator)


# -----------------------------------------------------------------------------
# Numeric continuous approximations
# -----------------------------------------------------------------------------

def numeric_normalization(
    density: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    num: int = 10001,
) -> float:
    """
    Numerically approximate ∫ f(x) dx using the trapezoidal rule.
    """
    if num < 2:
        raise ValueError("num must be at least 2.")
    x = np.linspace(lower, upper, num)
    y = np.asarray(density(x), dtype=float)
    if np.any(y < 0):
        raise ValueError("Density must be nonnegative on the sampled grid.")
    return _trapz(y, x)


def numeric_expectation(
    observable: Callable[[np.ndarray], np.ndarray],
    density: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    num: int = 10001,
    normalized: bool = False,
) -> float:
    """
    Numerically approximate E[g(x)] using the trapezoidal rule.
    """
    if num < 2:
        raise ValueError("num must be at least 2.")
    x = np.linspace(lower, upper, num)
    f = np.asarray(density(x), dtype=float)
    g = np.asarray(observable(x), dtype=float)

    if np.any(f < 0):
        raise ValueError("Density must be nonnegative on the sampled grid.")

    numerator = _trapz(g * f, x)
    if normalized:
        return float(numerator)

    denominator = _trapz(f, x)
    if denominator == 0:
        raise ValueError("Density integral is zero; expectation is undefined.")
    return float(numerator / denominator)


def numeric_average(
    density: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    num: int = 10001,
    normalized: bool = False,
) -> float:
    """
    Numerically approximate <x> for a continuous density.
    """
    return numeric_expectation(lambda x: x, density, lower, upper, num=num, normalized=normalized)


def sample_continuous_density(
    density,
    variable=None,
    lower: float = 0.0,
    upper: float = 1.0,
    num: int = 1000,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a continuous density on a numeric grid.

    `density` can be a callable or a SymPy expression. If a SymPy expression is
    used, `variable` must be provided.
    """
    if num < 2:
        raise ValueError("num must be at least 2.")

    x = np.linspace(lower, upper, num)
    y = np.asarray(_eval_function_or_expr(density, variable, x), dtype=float)

    if normalize:
        area = _trapz(y, x)
        if area == 0:
            raise ValueError("Sampled density has zero area; cannot normalize.")
        y = y / area

    return x, y


# -----------------------------------------------------------------------------
# Discrete distributions
# -----------------------------------------------------------------------------

def normalize_weights(weights: Iterable[Any]) -> np.ndarray:
    """
    Normalize discrete weights so that they sum to 1.
    """
    w = _as_object_array(weights)
    _validate_nonnegative_numeric(w)
    total = _safe_sum(w)
    if total == 0:
        raise ValueError("Sum of weights is zero; cannot normalize.")
    return np.asarray([simplify(item / total) for item in w], dtype=object)


def discrete_expectation(
    values: Iterable[Any],
    weights: Iterable[Any],
    normalized: bool = False,
):
    """
    Compute E[x] for a discrete distribution.

    Parameters
    ----------
    values : iterable
        Values x_i.
    weights : iterable
        Weights or probabilities w_i.
    normalized : bool
        If False, divide by sum(weights). If True, assume sum(weights) = 1.
    """
    x = _as_object_array(values)
    w = _as_object_array(weights)

    if x.shape != w.shape:
        raise ValueError("values and weights must have the same shape.")

    _validate_nonnegative_numeric(w)

    numerator = _safe_sum(x * w)
    if normalized:
        return simplify(numerator)

    denominator = _safe_sum(w)
    if denominator == 0:
        raise ValueError("Sum of weights is zero; expectation is undefined.")

    return simplify(numerator / denominator)


def discrete_average(values, weights=None, normalized=False):
    """
    Compute the discrete expectation value of x.

    Parameters
    ----------
    values : iterable
        Discrete values x_i.

    weights : iterable, callable, or None
        If iterable, interpreted as weights w_i.
        If callable, interpreted as a function f(x_i).
        If None, all values receive equal weight.

    normalized : bool
        If False, weights are normalized internally.
        If True, weights are assumed to already sum to 1.

    Returns
    -------
    float or symbolic expression
        Weighted average:

            <x> = sum_i x_i w_i / sum_i w_i
    """
    import numpy as np

    from relativity.utils import is_symbolic, simplify

    x = np.asarray(list(values), dtype=object)

    if weights is None:
        w = np.ones(len(x), dtype=object)

    elif callable(weights):
        w = np.asarray([weights(xi) for xi in x], dtype=object)

    else:
        w = np.asarray(list(weights), dtype=object)

    if len(x) != len(w):
        raise ValueError("values and weights must have the same length.")

    if np.any([wi < 0 for wi in w if not is_symbolic(wi)]):
        raise ValueError("weights must be nonnegative.")

    total_weight = sum(w)

    if not is_symbolic(total_weight) and total_weight == 0:
        raise ValueError("sum of weights cannot be zero.")

    if normalized:
        return simplify(sum(xi * wi for xi, wi in zip(x, w)))

    return simplify(sum(xi * wi for xi, wi in zip(x, w)) / total_weight)

def discrete_mean(values, weights=None, normalized=False):
    """
    Alias for discrete_average().
    """
    return discrete_average(values, weights=weights, normalized=normalized)


def discrete_average_from_function(values, function, normalized=False):
    """
    Compute a discrete average using weights generated by function(x).
    """
    return discrete_average(values, weights=function, normalized=normalized)


def discrete_moment(
    order: int,
    values: Iterable[Any],
    weights: Iterable[Any],
    center=None,
    normalized: bool = False,
):
    """
    Compute a discrete moment.

    If center is None, computes <x^order>.
    If center is given, computes <(x - center)^order>.
    """
    if order < 0:
        raise ValueError("Moment order must be nonnegative.")

    x = _as_object_array(values)
    observable = x**order if center is None else (x - center)**order
    return discrete_expectation(observable, weights, normalized=normalized)


def discrete_variance(values: Iterable[Any], weights: Iterable[Any], normalized: bool = False):
    """
    Compute Var(x) = <x²> - <x>² for a discrete distribution.
    """
    mean = discrete_average(values, weights, normalized=normalized)
    second = discrete_moment(2, values, weights, normalized=normalized)
    return simplify(second - mean**2)


def discrete_std(values: Iterable[Any], weights: Iterable[Any], normalized: bool = False):
    """
    Compute the standard deviation for a discrete distribution.
    """
    var = discrete_variance(values, weights, normalized=normalized)
    if is_symbolic(var):
        _require_sympy()
        return simplify(sp.sqrt(var))
    return math.sqrt(float(var))


def discrete_probabilities(values: Iterable[Any], weights: Iterable[Any]) -> dict:
    """
    Return a dictionary mapping each value to its normalized probability.
    """
    x = _as_object_array(values)
    p = normalize_weights(weights)
    return {x[i]: p[i] for i in range(len(x))}


def discrete_cdf(values: Iterable[Any], weights: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return sorted values and cumulative probabilities.

    This function is intended for numeric values because sorting symbolic
    values is generally ambiguous.
    """
    x = _as_float_array(values)
    p = np.asarray(normalize_weights(weights), dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    p_sorted = p[order]
    return x_sorted, np.cumsum(p_sorted)


def discrete_probability_between(
    values: Iterable[Any],
    weights: Iterable[Any],
    lower=None,
    upper=None,
    inclusive: bool = True,
):
    """
    Compute P(lower <= x <= upper) for a discrete distribution.

    If lower or upper is None, that side is unbounded.
    """
    x = _as_object_array(values)
    p = normalize_weights(weights)

    total = 0
    for xi, pi in zip(x, p):
        if is_symbolic(xi):
            raise ValueError("Cannot compare symbolic values in probability_between.")

        xi_float = float(xi)
        ok_lower = True if lower is None else (xi_float >= lower if inclusive else xi_float > lower)
        ok_upper = True if upper is None else (xi_float <= upper if inclusive else xi_float < upper)

        if ok_lower and ok_upper:
            total += pi

    return simplify(total)


def values_from_range(start: float, stop: float, step: float) -> np.ndarray:
    """
    Create a discrete grid including the final point when it lands on the grid.

    Useful for exercises comparing Δx = 1, Δx = 5, etc.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    n = int(round((stop - start) / step))
    values = np.asarray([start + i * step for i in range(n + 1)], dtype=float)
    if values[-1] > stop + 1e-12:
        values = values[:-1]
    return values


# -----------------------------------------------------------------------------
# Exercise-oriented summaries
# -----------------------------------------------------------------------------

def continuous_distribution_summary(
    density,
    variable,
    lower,
    upper,
    normalized: bool = False,
) -> dict:
    """
    Return normalization, mean, second moment, variance and std for a density.
    """
    integral = 1 if normalized else continuous_integral(density, variable, lower, upper)
    mean = continuous_average(density, variable, lower, upper, normalized=normalized)
    second = continuous_moment(2, density, variable, lower, upper, normalized=normalized)
    variance = simplify(second - mean**2)
    std = continuous_std(density, variable, lower, upper, normalized=normalized)

    return {
        "integral": simplify(integral),
        "normalization_constant": simplify(1 / integral) if integral != 0 else None,
        "mean": mean,
        "second_moment": second,
        "variance": variance,
        "std": std,
    }


def discrete_distribution_summary(
    values: Iterable[Any],
    weights: Iterable[Any],
    normalized: bool = False,
) -> dict:
    """
    Return normalization, mean, second moment, variance and std for a discrete distribution.
    """
    x = _as_object_array(values)
    w = _as_object_array(weights)
    total = _safe_sum(w) if not normalized else 1
    mean = discrete_average(x, w, normalized=normalized)
    second = discrete_moment(2, x, w, normalized=normalized)
    variance = simplify(second - mean**2)
    std = discrete_std(x, w, normalized=normalized)

    return {
        "values": x,
        "weights": w,
        "weight_sum": simplify(total),
        "probabilities": normalize_weights(w) if not normalized else w,
        "mean": mean,
        "second_moment": second,
        "variance": variance,
        "std": std,
    }


def compare_continuous_and_discrete_averages(
    density,
    variable,
    lower,
    upper,
    steps: Sequence[float],
) -> dict:
    """
    Compare the continuous mean with discrete-grid approximations.

    This is tailored for exercises where x is continuous first and then
    restricted to grids such as Δx = 1 or Δx = 5.
    """
    _require_sympy()
    continuous_mean = continuous_average(density, variable, lower, upper, normalized=False)

    f_numeric = sp.lambdify(variable, density, modules="numpy")
    comparisons = {}

    for step in steps:
        grid = values_from_range(float(lower), float(upper), float(step))
        weights = f_numeric(grid)
        mean = discrete_average(grid, weights, normalized=False)
        comparisons[step] = {
            "values": grid,
            "weights": np.asarray(weights, dtype=float),
            "mean": mean,
            "difference_from_continuous": simplify(mean - continuous_mean),
        }

    return {
        "continuous_mean": continuous_mean,
        "discrete": comparisons,
    }


def tarea3_exercise10_summary():
    """
    Summary for the Tarea 3 probability exercise:

        f(x) = (1/10) (10 - x)^2, 0 <= x <= 10.

    Computes the continuous average and the discrete averages for Δx = 1 and
    Δx = 5.
    """
    _require_sympy()
    x = sp.Symbol("x", real=True)
    f = sp.Rational(1, 10) * (10 - x)**2

    return compare_continuous_and_discrete_averages(
        density=f,
        variable=x,
        lower=0,
        upper=10,
        steps=[1, 5],
    )


__all__ = [
    "continuous_integral",
    "normalization_constant_continuous",
    "normalize_continuous_density",
    "continuous_expectation",
    "continuous_average",
    "continuous_moment",
    "continuous_variance",
    "continuous_std",
    "continuous_probability",
    "continuous_cdf",
    "numeric_normalization",
    "numeric_expectation",
    "numeric_average",
    "sample_continuous_density",
    "normalize_weights",
    "discrete_expectation",
    "discrete_average",
    "discrete_average_from_function",
    "discrete_moment",
    "discrete_variance",
    "discrete_std",
    "discrete_probabilities",
    "discrete_cdf",
    "discrete_probability_between",
    "values_from_range",
    "continuous_distribution_summary",
    "discrete_distribution_summary",
    "compare_continuous_and_discrete_averages",
    "tarea3_exercise10_summary",
]
