"""Plot styling helpers for the relativity package.

This module intentionally contains no physics.  It only centralizes small
Matplotlib style utilities used by the plotting modules.
"""

from __future__ import annotations

from contextlib import contextmanager


def require_matplotlib():
    """Import Matplotlib lazily and return ``matplotlib.pyplot``.

    The plotting package can be imported on systems without Matplotlib until a
    plotting function is actually called.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "Matplotlib is required for relativity.plotting. "
            "Install it with: pip install matplotlib"
        ) from exc
    return plt


def set_default_style(
    *,
    grid: bool = True,
    latex: bool = False,
    font_size: float = 11,
    figure_size: tuple[float, float] = (7.0, 5.0),
):
    """Apply a clean default style for plots.

    Parameters
    ----------
    grid:
        Whether axes should show a grid by default.
    latex:
        Whether Matplotlib should render text using LaTeX.  This requires a
        working LaTeX installation; keep it ``False`` for maximum portability.
    font_size:
        Base font size.
    figure_size:
        Default figure size in inches.
    """
    plt = require_matplotlib()
    plt.rcParams.update(
        {
            "figure.figsize": figure_size,
            "font.size": font_size,
            "axes.grid": grid,
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "text.usetex": latex,
        }
    )


def use_latex_labels(enable: bool = True):
    """Turn Matplotlib LaTeX text rendering on or off."""
    plt = require_matplotlib()
    plt.rcParams["text.usetex"] = bool(enable)


@contextmanager
def plot_style(**rc_params):
    """Temporarily apply Matplotlib rcParams inside a ``with`` block."""
    plt = require_matplotlib()
    with plt.rc_context(rc=rc_params):
        yield


def get_axis(ax=None, *, figsize: tuple[float, float] | None = None):
    """Return ``(fig, ax)`` using an existing axis or creating a new one."""
    plt = require_matplotlib()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def apply_axes_style(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    equal: bool = False,
    legend: bool = False,
):
    """Apply labels, title, grid, aspect ratio and optional legend to an axis."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.35)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
    return ax


def finalize_plot(
    fig,
    ax,
    *,
    show: bool = False,
    tight_layout: bool = True,
    return_fig_ax: bool = True,
):
    """Finalize a plot and optionally show it.

    By default this returns ``(fig, ax)`` so users can keep customizing the
    figure.  Set ``show=True`` for notebook/script convenience.
    """
    if tight_layout:
        fig.tight_layout()
    if show:
        require_matplotlib().show()
    if return_fig_ax:
        return fig, ax
    return ax
