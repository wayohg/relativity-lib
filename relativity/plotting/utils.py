"""Data-preparation helpers for relativity plotting.

The functions here convert Events, Worldlines, lists and arrays into numeric
coordinates that Matplotlib can draw.  They do not create figures themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

try:  # SymPy is already used by the main library, but keep this module robust.
    import sympy as sp
except ImportError:  # pragma: no cover - unlikely in this project
    sp = None


_AXIS_ALIASES = {
    "t": "t",
    "time": "t",
    "ct": "ct",
    "x": "x",
    "y": "y",
    "z": "z",
    "r0": "x",
    "r1": "y",
    "r2": "z",
}


def _library_is_symbolic(value: Any) -> bool:
    try:
        from relativity.utils import is_symbolic as _is_symbolic

        return bool(_is_symbolic(value))
    except Exception:
        if sp is None:
            return False
        if isinstance(value, sp.Basic):
            return True
        if isinstance(value, np.ndarray):
            return any(_library_is_symbolic(v) for v in value.ravel())
        if isinstance(value, (list, tuple, set)):
            return any(_library_is_symbolic(v) for v in value)
        return False


def is_symbolic(value: Any) -> bool:
    """Return True when a value contains symbolic SymPy objects."""
    return _library_is_symbolic(value)


def as_numeric(value: Any, substitutions: dict | None = None) -> float:
    """Convert one scalar to ``float``.

    Symbolic values must either be fully numeric or be made numeric with
    ``substitutions``.
    """
    if substitutions and sp is not None and isinstance(value, sp.Basic):
        value = value.subs(substitutions)
    try:
        return float(value)
    except Exception as exc:
        raise TypeError(
            f"Cannot convert {value!r} to float. "
            "Pass substitutions={symbol: value} for symbolic data."
        ) from exc


def as_numeric_array(data: Any, substitutions: dict | None = None) -> np.ndarray:
    """Convert list/array-like data to a numeric ``numpy.ndarray``."""
    array = np.array(data, dtype=object)
    converter = np.vectorize(lambda x: as_numeric(x, substitutions), otypes=[float])
    return converter(array)


def axis_name(axis: str) -> str:
    """Normalize a coordinate-axis name."""
    key = str(axis).strip().lower()
    if key not in _AXIS_ALIASES:
        raise ValueError(
            f"Unknown axis {axis!r}. Expected one of: "
            f"{', '.join(sorted(_AXIS_ALIASES))}."
        )
    return _AXIS_ALIASES[key]


def event_component(event: Any, axis: str, *, c: float | None = None) -> Any:
    """Extract one component from an Event-like object.

    Supported objects include ``physics.Event`` instances, mappings with keys
    like ``t``/``x``/``y``/``z``, and sequences ``(t, x, y, z)``.
    """
    axis = axis_name(axis)

    if isinstance(event, dict):
        if axis == "ct":
            if "ct" in event:
                return event["ct"]
            if "t" not in event:
                raise KeyError("Event mapping needs 'ct' or 't' for ct axis.")
            c_value = c if c is not None else event.get("c", 1.0)
            return c_value * event["t"]
        return event[axis]

    if axis == "ct":
        if hasattr(event, "fourvec") and hasattr(event.fourvec, "ct"):
            return event.fourvec.ct
        if hasattr(event, "ct"):
            return event.ct
        if hasattr(event, "t"):
            c_value = c if c is not None else getattr(event, "c", 1.0)
            return c_value * event.t
    elif hasattr(event, axis):
        return getattr(event, axis)

    if hasattr(event, "r") and axis in {"x", "y", "z"}:
        index = {"x": 0, "y": 1, "z": 2}[axis]
        return event.r[index]

    seq = list(event)
    if axis == "t":
        return seq[0]
    if axis == "ct":
        c_value = 1.0 if c is None else c
        return c_value * seq[0]
    return seq[{"x": 1, "y": 2, "z": 3}[axis]]


def event_xy(
    event: Any,
    *,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
) -> tuple[float, float]:
    """Return numeric ``(x, y)`` coordinates for an Event-like object."""
    x = event_component(event, x_axis, c=c)
    y = event_component(event, y_axis, c=c)
    return as_numeric(x, substitutions), as_numeric(y, substitutions)


def iter_events(obj: Any):
    """Yield events from a Worldline-like object, iterable, or single event."""
    if hasattr(obj, "events"):
        yield from obj.events
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
        yield from obj
    else:
        yield obj


def worldline_xy(
    worldline_or_events: Any,
    *,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric arrays ``x, y`` from a Worldline-like object."""
    points = [
        event_xy(event, x_axis=x_axis, y_axis=y_axis, c=c, substitutions=substitutions)
        for event in iter_events(worldline_or_events)
    ]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    x, y = zip(*points)
    return np.array(x, dtype=float), np.array(y, dtype=float)


def beta_grid(
    beta_min: float = 0.0,
    beta_max: float = 0.999,
    n: int = 500,
    *,
    include_endpoint: bool = True,
) -> np.ndarray:
    """Return a valid grid of beta values with ``|beta| < 1``."""
    if beta_min <= -1 or beta_max >= 1:
        raise ValueError("beta values must satisfy -1 < beta < 1.")
    if n < 2:
        raise ValueError("n must be at least 2.")
    return np.linspace(beta_min, beta_max, n, endpoint=include_endpoint)


def finite_limits(*arrays: Any, margin: float = 0.05) -> tuple[float, float]:
    """Compute finite plotting limits with a fractional margin."""
    data = []
    for array in arrays:
        arr = np.asarray(array, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            data.append(arr)
    if not data:
        return -1.0, 1.0
    values = np.concatenate(data)
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        delta = 1.0 if lo == 0 else abs(lo) * margin
        return lo - delta, hi + delta
    delta = (hi - lo) * margin
    return lo - delta, hi + delta
