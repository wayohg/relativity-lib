"""Spacetime diagram plotting tools for special relativity.

Default diagrams use the horizontal axis ``x`` and the vertical axis ``ct``.
With this convention, light rays through the origin appear as 45-degree lines
when ``c=1`` or when using actual ``ct`` values.
"""

from __future__ import annotations

import numpy as np

from relativity.plotting.style import apply_axes_style, finalize_plot, get_axis
from relativity.plotting.utils import event_xy, finite_limits, worldline_xy


def _numeric_beta(beta: float) -> float:
    beta = float(beta)
    if abs(beta) >= 1:
        raise ValueError("beta must satisfy |beta| < 1.")
    return beta


def plot_light_cone(
    *,
    ax=None,
    origin: tuple[float, float] = (0.0, 0.0),
    extent: float = 1.0,
    use_ct: bool = True,
    c: float = 1.0,
    label: str | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot the two light-cone branches through an origin.

    Parameters
    ----------
    origin:
        ``(x0, ct0)`` when ``use_ct=True``.  If ``use_ct=False``, interpret the
        vertical coordinate as ``t0``.
    extent:
        Half-width of the plotted cone in the horizontal coordinate.
    use_ct:
        If True, plot in the ``x``-``ct`` plane.  If False, plot in ``x``-``t``.
    c:
        Speed of light used only when ``use_ct=False``.
    """
    fig, ax = get_axis(ax)
    x0, y0 = origin
    dx = np.linspace(-float(extent), float(extent), 400)

    if use_ct:
        y_plus = y0 + dx
        y_minus = y0 - dx
        ylabel = r"$ct$"
    else:
        y_plus = y0 + dx / c
        y_minus = y0 - dx / c
        ylabel = r"$t$"

    kwargs = {"linestyle": "--", "linewidth": 1.2}
    kwargs.update(line_kwargs)
    ax.plot(x0 + dx, y_plus, label=label, **kwargs)
    ax.plot(x0 + dx, y_minus, **kwargs)
    apply_axes_style(ax, xlabel=r"$x$", ylabel=ylabel, equal=use_ct)
    return finalize_plot(fig, ax, show=show)


def plot_event(
    event,
    *,
    ax=None,
    label: str | None = None,
    annotate: bool = True,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
    show: bool = False,
    **scatter_kwargs,
):
    """Plot one Event-like object on a spacetime diagram."""
    fig, ax = get_axis(ax)
    x, y = event_xy(event, x_axis=x_axis, y_axis=y_axis, c=c, substitutions=substitutions)
    kwargs = {"s": 35}
    kwargs.update(scatter_kwargs)
    ax.scatter([x], [y], label=None if annotate else label, **kwargs)
    if annotate and label:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    apply_axes_style(
        ax,
        xlabel=rf"${x_axis}$" if len(x_axis) <= 2 else x_axis,
        ylabel=rf"${y_axis}$" if len(y_axis) <= 2 else y_axis,
        equal=(y_axis == "ct"),
        legend=bool(label and not annotate),
    )
    return finalize_plot(fig, ax, show=show)


def plot_events(
    events,
    *,
    ax=None,
    labels=None,
    annotate: bool = True,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
    show: bool = False,
    **scatter_kwargs,
):
    """Plot several Event-like objects."""
    fig, ax = get_axis(ax)
    labels = list(labels) if labels is not None else [None] * len(events)
    points = [
        event_xy(event, x_axis=x_axis, y_axis=y_axis, c=c, substitutions=substitutions)
        for event in events
    ]
    if points:
        xs, ys = zip(*points)
        kwargs = {"s": 35}
        kwargs.update(scatter_kwargs)
        ax.scatter(xs, ys, **kwargs)
        if annotate:
            for label, x, y in zip(labels, xs, ys):
                if label:
                    ax.annotate(str(label), (x, y), textcoords="offset points", xytext=(5, 5))
    apply_axes_style(
        ax,
        xlabel=rf"${x_axis}$" if len(x_axis) <= 2 else x_axis,
        ylabel=rf"${y_axis}$" if len(y_axis) <= 2 else y_axis,
        equal=(y_axis == "ct"),
    )
    return finalize_plot(fig, ax, show=show)


def plot_interval(
    event_a,
    event_b,
    *,
    ax=None,
    label: str | None = None,
    x_axis: str = "x",
    y_axis: str = "ct",
    c: float | None = None,
    substitutions: dict | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Draw the segment connecting two events."""
    fig, ax = get_axis(ax)
    xa, ya = event_xy(event_a, x_axis=x_axis, y_axis=y_axis, c=c, substitutions=substitutions)
    xb, yb = event_xy(event_b, x_axis=x_axis, y_axis=y_axis, c=c, substitutions=substitutions)
    kwargs = {"linewidth": 1.5}
    kwargs.update(line_kwargs)
    ax.plot([xa, xb], [ya, yb], label=label, **kwargs)
    apply_axes_style(
        ax,
        xlabel=rf"${x_axis}$" if len(x_axis) <= 2 else x_axis,
        ylabel=rf"${y_axis}$" if len(y_axis) <= 2 else y_axis,
        equal=(y_axis == "ct"),
        legend=bool(label),
    )
    return finalize_plot(fig, ax, show=show)


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
    """Plot a Worldline-like object or a list of events."""
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
        xlabel=rf"${x_axis}$" if len(x_axis) <= 2 else x_axis,
        ylabel=rf"${y_axis}$" if len(y_axis) <= 2 else y_axis,
        equal=(y_axis == "ct"),
        legend=bool(label),
    )
    return finalize_plot(fig, ax, show=show)


def plot_reference_frame_axes(
    beta: float,
    *,
    ax=None,
    origin: tuple[float, float] = (0.0, 0.0),
    extent: float = 1.0,
    label: str = "S'",
    show: bool = False,
    **line_kwargs,
):
    """Plot the tilted ``x'`` and ``ct'`` axes for a frame moving at beta.

    This function assumes an ``x``-``ct`` diagram.  The ``ct'`` axis satisfies
    ``x = beta ct`` and the ``x'`` axis satisfies ``ct = beta x``.
    """
    beta = _numeric_beta(beta)
    fig, ax = get_axis(ax)
    x0, ct0 = origin
    s = np.linspace(-float(extent), float(extent), 300)
    kwargs = {"linewidth": 1.2}
    kwargs.update(line_kwargs)

    # x' axis: ct - ct0 = beta (x - x0)
    ax.plot(x0 + s, ct0 + beta * s, label=rf"${label}: x'$", **kwargs)

    # ct' axis: x - x0 = beta (ct - ct0)
    ct_values = ct0 + s
    x_values = x0 + beta * s
    ax.plot(x_values, ct_values, label=rf"${label}: ct'$", **kwargs)

    apply_axes_style(ax, xlabel=r"$x$", ylabel=r"$ct$", equal=True, legend=True)
    return finalize_plot(fig, ax, show=show)


def plot_spacetime_diagram(
    *,
    events=None,
    event_labels=None,
    worldlines=None,
    worldline_labels=None,
    light_cone: bool = True,
    beta_axes: list[float] | tuple[float, ...] | None = None,
    ax=None,
    extent: float | None = None,
    c: float | None = None,
    substitutions: dict | None = None,
    title: str | None = "Spacetime diagram",
    show: bool = False,
):
    """Create a basic spacetime diagram from events and worldlines."""
    fig, ax = get_axis(ax)
    events = list(events) if events is not None else []
    worldlines = list(worldlines) if worldlines is not None else []
    worldline_labels = (
        list(worldline_labels) if worldline_labels is not None else [None] * len(worldlines)
    )

    all_x = []
    all_y = []

    for worldline, label in zip(worldlines, worldline_labels):
        xs, ys = worldline_xy(worldline, c=c, substitutions=substitutions)
        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())
        ax.plot(xs, ys, label=label, linewidth=1.8)

    if events:
        labels = list(event_labels) if event_labels is not None else [None] * len(events)
        points = [event_xy(event, c=c, substitutions=substitutions) for event in events]
        xs, ys = zip(*points)
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(xs, ys, s=35)
        for label, x, y in zip(labels, xs, ys):
            if label:
                ax.annotate(str(label), (x, y), textcoords="offset points", xytext=(5, 5))

    if extent is None:
        lo, hi = finite_limits(all_x, all_y, margin=0.12)
        extent = max(abs(lo), abs(hi), 1.0)

    if light_cone:
        plot_light_cone(ax=ax, extent=extent, show=False, linewidth=1.0, linestyle="--")

    if beta_axes:
        for beta in beta_axes:
            plot_reference_frame_axes(beta, ax=ax, extent=extent, show=False)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    apply_axes_style(
        ax,
        xlabel=r"$x$",
        ylabel=r"$ct$",
        title=title,
        equal=True,
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)
