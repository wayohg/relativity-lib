"""
relativity.plotting.kinematics
==============================

Visualization helpers for special-relativistic kinematics.

This module is intentionally numeric-oriented: plotting requires concrete
numeric values, even if the rest of the library supports symbolic work.

Main features
-------------
- gamma vs beta
- time dilation vs beta
- length contraction vs beta
- rapidity vs beta
- beta vs rapidity
- 1D relativistic velocity addition
- longitudinal Doppler factor
- longitudinal redshift / blueshift

All plotting functions return (fig, ax).

Example
-------
from relativity.plotting.kinematics import plot_gamma_vs_beta

fig, ax = plot_gamma_vs_beta(show=True)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Optional integration with plotting.style
# ---------------------------------------------------------------------

try:
    from .style import set_default_style, add_grid, format_axis
except Exception:
    def set_default_style() -> None:
        """Fallback style setup."""
        return None

    def add_grid(ax, which: str = "both", alpha: float = 0.3) -> None:
        """Fallback grid helper."""
        ax.grid(True, which=which, alpha=alpha)

    def format_axis(
        ax,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        legend: bool = True,
    ) -> None:
        """Fallback axis formatter."""
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _prepare_axes(
    ax=None,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Create or reuse an axes object."""
    set_default_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    return fig, ax


def _finalize_axes(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend: bool = True,
    grid: bool = True,
) -> None:
    """Apply common formatting."""
    format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend)

    if grid:
        add_grid(ax)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)


def _show_if_requested(show: bool) -> None:
    """Show plot if requested."""
    if show:
        plt.show()


def _beta_array(
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
) -> np.ndarray:
    """Return a safe beta array."""
    if beta_min < -0.999999999:
        raise ValueError("beta_min must be greater than -1.")
    if beta_max >= 1.0:
        raise ValueError("beta_max must be strictly less than 1.")
    if beta_min >= beta_max:
        raise ValueError("beta_min must be smaller than beta_max.")
    if num < 2:
        raise ValueError("num must be at least 2.")

    return np.linspace(beta_min, beta_max, num)


def _gamma_from_beta(beta) -> np.ndarray:
    """Lorentz factor gamma(beta)."""
    beta = np.asarray(beta, dtype=float)
    return 1.0 / np.sqrt(1.0 - beta**2)


def _rapidity_from_beta(beta) -> np.ndarray:
    """Rapidity eta(beta) = atanh(beta)."""
    beta = np.asarray(beta, dtype=float)
    return np.arctanh(beta)


def _beta_from_rapidity(eta) -> np.ndarray:
    """beta(eta) = tanh(eta)."""
    eta = np.asarray(eta, dtype=float)
    return np.tanh(eta)


def _velocity_addition_1d_beta(beta_u, beta_v) -> np.ndarray:
    """
    1D relativistic velocity addition in dimensionless form.

    Interprets beta_u and beta_v as dimensionless velocities u/c and v/c:
        beta = (beta_u + beta_v) / (1 + beta_u * beta_v)
    """
    beta_u = np.asarray(beta_u, dtype=float)
    beta_v = np.asarray(beta_v, dtype=float)

    denom = 1.0 + beta_u * beta_v
    return (beta_u + beta_v) / denom


def _doppler_factor_longitudinal(beta, approaching: bool = False) -> np.ndarray:
    """
    Longitudinal Doppler factor.

    approaching=False:
        D = sqrt((1 - beta) / (1 + beta))   -> source receding, redshift

    approaching=True:
        D = sqrt((1 + beta) / (1 - beta))   -> source approaching, blueshift
    """
    beta = np.asarray(beta, dtype=float)

    if approaching:
        return np.sqrt((1.0 + beta) / (1.0 - beta))

    return np.sqrt((1.0 - beta) / (1.0 + beta))


def _redshift_from_beta(beta) -> np.ndarray:
    """
    Longitudinal redshift for a receding source.

    1 + z = sqrt((1 + beta) / (1 - beta))
    """
    beta = np.asarray(beta, dtype=float)
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


def _blueshift_from_beta(beta) -> np.ndarray:
    """
    Longitudinal blueshift represented as a negative z-like quantity
    for an approaching source:

    1 + z = sqrt((1 - beta) / (1 + beta))
    """
    beta = np.asarray(beta, dtype=float)
    return np.sqrt((1.0 - beta) / (1.0 + beta)) - 1.0


# ---------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------

def plot_gamma_vs_beta(
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
    ax=None,
    show: bool = False,
    annotate_beta: Optional[float] = None,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot the Lorentz factor gamma as a function of beta = v/c.
    """
    beta = _beta_array(beta_min, beta_max, num)
    gamma = _gamma_from_beta(beta)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(beta, gamma, label=r"$\gamma(\beta)=\frac{1}{\sqrt{1-\beta^2}}$")

    if annotate_beta is not None:
        if not (-1.0 < annotate_beta < 1.0):
            raise ValueError("annotate_beta must satisfy -1 < beta < 1.")
        g = _gamma_from_beta(np.array([annotate_beta]))[0]
        ax.scatter([annotate_beta], [g], zorder=3)
        ax.annotate(
            fr"$\beta={annotate_beta:.3f}$" + "\n" + fr"$\gamma={g:.3f}$",
            xy=(annotate_beta, g),
            xytext=(10, 10),
            textcoords="offset points",
        )

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=r"$\gamma$",
        title="Lorentz Factor vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_time_dilation(
    proper_time: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot dilated time Δt = gamma * Δτ as a function of beta.
    """
    beta = _beta_array(beta_min, beta_max, num)
    gamma = _gamma_from_beta(beta)
    lab_time = gamma * float(proper_time)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(beta, lab_time, label=fr"$\Delta t = \gamma \Delta \tau,\ \Delta \tau={proper_time}$")
    ax.axhline(proper_time, linestyle="--", alpha=0.7, label=r"Proper time $\Delta \tau$")

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=r"$\Delta t$",
        title="Time Dilation vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_length_contraction(
    proper_length: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot contracted length L = L0 / gamma as a function of beta.
    """
    beta = _beta_array(beta_min, beta_max, num)
    gamma = _gamma_from_beta(beta)
    length = float(proper_length) / gamma

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(beta, length, label=fr"$L = L_0/\gamma,\ L_0={proper_length}$")
    ax.axhline(proper_length, linestyle="--", alpha=0.7, label=r"Proper length $L_0$")

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=r"$L$",
        title="Length Contraction vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_rapidity_vs_beta(
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    num: int = 500,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot rapidity eta = atanh(beta) as a function of beta.
    """
    beta = _beta_array(beta_min, beta_max, num)
    eta = _rapidity_from_beta(beta)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(beta, eta, label=r"$\eta = \operatorname{artanh}(\beta)$")
    ax.axhline(0.0, linestyle="--", alpha=0.6)
    ax.axvline(0.0, linestyle="--", alpha=0.6)

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=r"$\eta$",
        title="Rapidity vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_beta_vs_rapidity(
    eta_min: float = -3.0,
    eta_max: float = 3.0,
    num: int = 500,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot beta = tanh(eta) as a function of rapidity eta.
    """
    if eta_min >= eta_max:
        raise ValueError("eta_min must be smaller than eta_max.")
    if num < 2:
        raise ValueError("num must be at least 2.")

    eta = np.linspace(eta_min, eta_max, num)
    beta = _beta_from_rapidity(eta)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(eta, beta, label=r"$\beta = \tanh(\eta)$")
    ax.axhline(0.0, linestyle="--", alpha=0.6)
    ax.axvline(0.0, linestyle="--", alpha=0.6)

    _finalize_axes(
        ax,
        xlabel=r"$\eta$",
        ylabel=r"$\beta = v/c$",
        title="Beta vs Rapidity",
        xlim=(eta_min, eta_max),
        ylim=(-1.02, 1.02),
    )
    _show_if_requested(show)
    return fig, ax


def plot_velocity_addition_1d(
    frame_beta: float = 0.5,
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    num: int = 500,
    compare_classical: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot 1D relativistic velocity addition.

    Parameters
    ----------
    frame_beta : float
        Dimensionless velocity of one frame/object, v/c.
    compare_classical : bool
        If True, also plot the classical sum beta_u + frame_beta.
    """
    if not (-1.0 < frame_beta < 1.0):
        raise ValueError("frame_beta must satisfy -1 < beta < 1.")

    beta_u = _beta_array(beta_min, beta_max, num)
    beta_rel = _velocity_addition_1d_beta(beta_u, frame_beta)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(
        beta_u,
        beta_rel,
        label=fr"Relativistic: $\beta = \frac{{\beta_u+\beta_v}}{{1+\beta_u\beta_v}}$, $\beta_v={frame_beta}$",
    )

    if compare_classical:
        beta_classical = beta_u + frame_beta
        ax.plot(beta_u, beta_classical, linestyle="--", alpha=0.8, label="Classical sum")

    ax.axhline(1.0, linestyle=":", alpha=0.6)
    ax.axhline(-1.0, linestyle=":", alpha=0.6)

    _finalize_axes(
        ax,
        xlabel=r"Input $\beta_u$",
        ylabel=r"Composed $\beta$",
        title="1D Relativistic Velocity Addition",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_longitudinal_doppler_factor(
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
    show_approaching: bool = True,
    show_receding: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot longitudinal Doppler factor vs beta.

    Approaching:
        D = sqrt((1 + beta)/(1 - beta))

    Receding:
        D = sqrt((1 - beta)/(1 + beta))
    """
    beta = _beta_array(beta_min, beta_max, num)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)

    if show_approaching:
        d_app = _doppler_factor_longitudinal(beta, approaching=True)
        ax.plot(beta, d_app, label="Approaching source")

    if show_receding:
        d_rec = _doppler_factor_longitudinal(beta, approaching=False)
        ax.plot(beta, d_rec, label="Receding source")

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel="Doppler factor",
        title="Longitudinal Doppler Factor vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_redshift_vs_beta(
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    num: int = 500,
    include_blueshift: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """
    Plot longitudinal redshift z vs beta.

    Receding:
        1 + z = sqrt((1 + beta)/(1 - beta))

    Approaching curve is shown as a negative z-like quantity if
    include_blueshift=True.
    """
    beta = _beta_array(beta_min, beta_max, num)
    z_red = _redshift_from_beta(beta)

    fig, ax = _prepare_axes(ax=ax, figsize=figsize)
    ax.plot(beta, z_red, label="Receding source (redshift)")

    if include_blueshift:
        z_blue = _blueshift_from_beta(beta)
        ax.plot(beta, z_blue, label="Approaching source (blueshift)")

    ax.axhline(0.0, linestyle="--", alpha=0.6)

    _finalize_axes(
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=r"$z$",
        title="Longitudinal Redshift / Blueshift vs Beta",
        xlim=(beta_min, beta_max),
    )
    _show_if_requested(show)
    return fig, ax


def plot_kinematics_summary(
    beta_max: float = 0.95,
    num: int = 400,
    proper_time: float = 1.0,
    proper_length: float = 1.0,
    figsize: Tuple[float, float] = (11.0, 8.0),
    show: bool = False,
):
    """
    Create a compact 2x2 summary figure:
    - gamma vs beta
    - time dilation vs beta
    - length contraction vs beta
    - rapidity vs beta
    """
    if beta_max >= 1.0:
        raise ValueError("beta_max must be strictly less than 1.")

    set_default_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_gamma_vs_beta(beta_min=0.0, beta_max=beta_max, num=num, ax=axes[0, 0], show=False)
    plot_time_dilation(
        proper_time=proper_time,
        beta_min=0.0,
        beta_max=beta_max,
        num=num,
        ax=axes[0, 1],
        show=False,
    )
    plot_length_contraction(
        proper_length=proper_length,
        beta_min=0.0,
        beta_max=beta_max,
        num=num,
        ax=axes[1, 0],
        show=False,
    )
    plot_rapidity_vs_beta(beta_min=0.0, beta_max=beta_max, num=num, ax=axes[1, 1], show=False)

    fig.suptitle("Special Relativity Kinematics Summary")
    fig.tight_layout()

    _show_if_requested(show)
    return fig, axes


__all__ = [
    "plot_gamma_vs_beta",
    "plot_time_dilation",
    "plot_length_contraction",
    "plot_rapidity_vs_beta",
    "plot_beta_vs_rapidity",
    "plot_velocity_addition_1d",
    "plot_longitudinal_doppler_factor",
    "plot_redshift_vs_beta",
    "plot_kinematics_summary",
]