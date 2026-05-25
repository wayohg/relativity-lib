"""Worldline plotting tools for special relativity.

This module visualizes already-computed events/worldlines.  It intentionally
keeps physics calculations minimal: finite differences are used only to prepare
velocity/acceleration curves from sampled events.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from relativity.plotting.style import apply_axes_style, finalize_plot, get_axis
from relativity.plotting.utils import (
    as_numeric,
    as_numeric_array,
    axis_name,
    event_component,
    finite_limits,
    iter_events,
    worldline_xy,
)


_SPATIAL_AXES = ("x", "y", "z")


def _label_for_axis(axis: str) -> str:
    axis = axis_name(axis)
    return rf"${axis}$" if len(axis) <= 2 else str(axis)


def _axis_list(axes: str | Sequence[str]) -> list[str]:
    if isinstance(axes, str):
        axes = [axes]
    result = [axis_name(axis) for axis in axes]
    invalid = [axis for axis in result if axis not in _SPATIAL_AXES]
    if invalid:
        raise ValueError("Worldline component plots expect spatial axes: 'x', 'y', 'z'.")
    return result


def _get_events(worldline_or_events: Any) -> list[Any]:
    events = list(iter_events(worldline_or_events))
    if not events:
        raise ValueError("The worldline has no events to plot.")
    return events


def _get_c(worldline_or_events: Any, c: float | None = None) -> float:
    if c is not None:
        return float(c)
    return float(getattr(worldline_or_events, "c", 1.0))


def _time_array(
    events: Sequence[Any],
    *,
    substitutions: dict | None = None,
) -> np.ndarray:
    return as_numeric_array(
        [event_component(event, "t") for event in events],
        substitutions=substitutions,
    )


def _position_array(
    events: Sequence[Any],
    *,
    substitutions: dict | None = None,
) -> np.ndarray:
    rows = []
    for event in events:
        rows.append([event_component(event, axis) for axis in _SPATIAL_AXES])
    return as_numeric_array(rows, substitutions=substitutions)


def _midpoints(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values[:-1] + values[1:])


def _segment_velocities(
    worldline_or_events: Any,
    *,
    substitutions: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment midpoint times and finite-difference velocities."""
    events = _get_events(worldline_or_events)
    if len(events) < 2:
        raise ValueError("At least two events are required to compute velocity.")

    t = _time_array(events, substitutions=substitutions)

    if hasattr(worldline_or_events, "velocities"):
        try:
            v = as_numeric_array(worldline_or_events.velocities(), substitutions=substitutions)
        except Exception:
            r = _position_array(events, substitutions=substitutions)
            v = np.diff(r, axis=0) / np.diff(t)[:, None]
    else:
        r = _position_array(events, substitutions=substitutions)
        v = np.diff(r, axis=0) / np.diff(t)[:, None]

    if v.ndim == 1:
        v = v.reshape(-1, 3)
    return _midpoints(t), v


def _segment_accelerations(
    worldline_or_events: Any,
    *,
    substitutions: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity-midpoint times and finite-difference accelerations."""
    tm, v = _segment_velocities(worldline_or_events, substitutions=substitutions)
    if len(tm) < 2:
        raise ValueError("At least three events are required to compute acceleration.")
    a = np.diff(v, axis=0) / np.diff(tm)[:, None]
    return _midpoints(tm), a


def _vector_norm(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(values * values, axis=1))


def plot_worldline(
    worldline_or_events,
    *,
    ax=None,
    label: str | None = None,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
    show_events: bool = False,
    show: bool = False,
    **line_kwargs,
):
    """Plot a worldline using arbitrary coordinate axes.

    The default is an ``x``-``ct`` spacetime diagram.  For ordinary position-time
    plots, use :func:`plot_position_time`.
    """
    fig, ax = get_axis(ax)
    xs, ys = worldline_xy(
        worldline_or_events,
        x_axis=x_axis,
        y_axis=y_axis,
        c=c,
        substitutions=substitutions,
    )
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(xs, ys, label=label, **kwargs)
    if show_events and len(xs):
        ax.scatter(xs, ys, s=20)
    apply_axes_style(
        ax,
        xlabel=_label_for_axis(x_axis),
        ylabel=_label_for_axis(y_axis),
        equal=(axis_name(y_axis) == "ct"),
        legend=bool(label),
    )
    return finalize_plot(fig, ax, show=show)


def plot_worldlines(
    worldlines,
    *,
    ax=None,
    labels=None,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
    show_events: bool = False,
    show: bool = False,
    **line_kwargs,
):
    """Plot several worldlines on the same axis."""
    fig, ax = get_axis(ax)
    worldlines = list(worldlines)
    labels = list(labels) if labels is not None else [None] * len(worldlines)

    for worldline, label in zip(worldlines, labels):
        xs, ys = worldline_xy(
            worldline,
            x_axis=x_axis,
            y_axis=y_axis,
            c=c,
            substitutions=substitutions,
        )
        kwargs = {"linewidth": 1.8}
        kwargs.update(line_kwargs)
        ax.plot(xs, ys, label=label, **kwargs)
        if show_events and len(xs):
            ax.scatter(xs, ys, s=20)

    apply_axes_style(
        ax,
        xlabel=_label_for_axis(x_axis),
        ylabel=_label_for_axis(y_axis),
        equal=(axis_name(y_axis) == "ct"),
        legend=any(label is not None for label in labels),
    )
    return finalize_plot(fig, ax, show=show)


def plot_position_time(
    worldline_or_events,
    *,
    ax=None,
    components: str | Sequence[str] = "x",
    labels: Sequence[str] | None = None,
    substitutions: dict | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot position components as functions of coordinate time."""
    events = _get_events(worldline_or_events)
    t = _time_array(events, substitutions=substitutions)
    r = _position_array(events, substitutions=substitutions)
    axes = _axis_list(components)
    labels = list(labels) if labels is not None else axes

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    for axis, label in zip(axes, labels):
        idx = _SPATIAL_AXES.index(axis)
        ax.plot(t, r[:, idx], label=label, **kwargs)

    apply_axes_style(
        ax,
        xlabel=r"$t$",
        ylabel="position",
        legend=len(axes) > 1 or any(label is not None for label in labels),
    )
    return finalize_plot(fig, ax, show=show)


def plot_velocity_time(
    worldline_or_events,
    *,
    ax=None,
    components: str | Sequence[str] = "x",
    labels: Sequence[str] | None = None,
    substitutions: dict | None = None,
    show_speed: bool = False,
    show: bool = False,
    **line_kwargs,
):
    """Plot velocity components, or speed, as functions of coordinate time."""
    tm, v = _segment_velocities(worldline_or_events, substitutions=substitutions)
    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)

    if show_speed:
        ax.plot(tm, _vector_norm(v), label="speed", **kwargs)
        legend = True
    else:
        axes = _axis_list(components)
        labels = list(labels) if labels is not None else [f"v_{axis}" for axis in axes]
        for axis, label in zip(axes, labels):
            idx = _SPATIAL_AXES.index(axis)
            ax.plot(tm, v[:, idx], label=label, **kwargs)
        legend = len(axes) > 1 or any(label is not None for label in labels)

    apply_axes_style(ax, xlabel=r"$t$", ylabel="velocity", legend=legend)
    return finalize_plot(fig, ax, show=show)


def plot_speed_time(
    worldline_or_events,
    *,
    ax=None,
    substitutions: dict | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Convenience wrapper for speed versus coordinate time."""
    return plot_velocity_time(
        worldline_or_events,
        ax=ax,
        substitutions=substitutions,
        show_speed=True,
        show=show,
        **line_kwargs,
    )


def plot_acceleration_time(
    worldline_or_events,
    *,
    ax=None,
    components: str | Sequence[str] = "x",
    labels: Sequence[str] | None = None,
    substitutions: dict | None = None,
    show_magnitude: bool = False,
    show: bool = False,
    **line_kwargs,
):
    """Plot finite-difference acceleration components versus coordinate time."""
    tm, a = _segment_accelerations(worldline_or_events, substitutions=substitutions)
    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)

    if show_magnitude:
        ax.plot(tm, _vector_norm(a), label="acceleration magnitude", **kwargs)
        legend = True
    else:
        axes = _axis_list(components)
        labels = list(labels) if labels is not None else [f"a_{axis}" for axis in axes]
        for axis, label in zip(axes, labels):
            idx = _SPATIAL_AXES.index(axis)
            ax.plot(tm, a[:, idx], label=label, **kwargs)
        legend = len(axes) > 1 or any(label is not None for label in labels)

    apply_axes_style(ax, xlabel=r"$t$", ylabel="acceleration", legend=legend)
    return finalize_plot(fig, ax, show=show)


def cumulative_proper_time(
    worldline_or_events,
    *,
    c: float | None = None,
    substitutions: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinate times and cumulative proper time for sampled events.

    Space-like segments are assigned ``nan`` after they occur because they do not
    have real timelike proper time.
    """
    events = _get_events(worldline_or_events)
    c_value = _get_c(worldline_or_events, c)
    t = _time_array(events, substitutions=substitutions)
    r = _position_array(events, substitutions=substitutions)

    tau = np.zeros(len(events), dtype=float)
    invalid = False
    for i in range(len(events) - 1):
        if invalid:
            tau[i + 1] = np.nan
            continue
        dt = t[i + 1] - t[i]
        dr = r[i + 1] - r[i]
        ds2 = c_value**2 * dt**2 - float(np.dot(dr, dr))
        if ds2 < 0:
            tau[i + 1] = np.nan
            invalid = True
        else:
            tau[i + 1] = tau[i] + np.sqrt(ds2) / c_value
    return t, tau


def plot_proper_time(
    worldline_or_events,
    *,
    ax=None,
    c: float | None = None,
    substitutions: dict | None = None,
    label: str | None = r"$\tau$",
    show: bool = False,
    **line_kwargs,
):
    """Plot cumulative proper time against coordinate time."""
    t, tau = cumulative_proper_time(
        worldline_or_events,
        c=c,
        substitutions=substitutions,
    )
    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(t, tau, label=label, **kwargs)
    apply_axes_style(ax, xlabel=r"$t$", ylabel=r"$\tau$", legend=bool(label))
    return finalize_plot(fig, ax, show=show)


def plot_worldline_summary(
    worldline_or_events,
    *,
    c: float | None = None,
    substitutions: dict | None = None,
    show: bool = False,
):
    """Create a compact four-panel summary of a sampled worldline.

    This is the only function in the module that creates multiple axes because
    its purpose is an overview dashboard: ``x(t)``, speed, acceleration magnitude,
    and cumulative proper time.
    """
    from relativity.plotting.style import require_matplotlib

    plt = require_matplotlib()
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0))
    axes = axs.ravel()

    plot_position_time(
        worldline_or_events,
        ax=axes[0],
        components="x",
        substitutions=substitutions,
        show=False,
    )
    axes[0].set_title("Position")

    plot_speed_time(
        worldline_or_events,
        ax=axes[1],
        substitutions=substitutions,
        show=False,
    )
    axes[1].set_title("Speed")

    try:
        plot_acceleration_time(
            worldline_or_events,
            ax=axes[2],
            show_magnitude=True,
            substitutions=substitutions,
            show=False,
        )
    except ValueError:
        axes[2].text(0.5, 0.5, "Need at least 3 events", ha="center", va="center")
        apply_axes_style(axes[2], title="Acceleration")

    plot_proper_time(
        worldline_or_events,
        ax=axes[3],
        c=c,
        substitutions=substitutions,
        show=False,
    )
    axes[3].set_title("Proper time")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


__all__ = [
    "plot_worldline",
    "plot_worldlines",
    "plot_position_time",
    "plot_velocity_time",
    "plot_speed_time",
    "plot_acceleration_time",
    "cumulative_proper_time",
    "plot_proper_time",
    "plot_worldline_summary",
]
