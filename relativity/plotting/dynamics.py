"""
relativity.plotting.dynamics
============================

Plotting helpers for special-relativistic dynamics.

This module is numeric-oriented: plotting requires concrete numeric values,
even if the rest of the library supports symbolic calculations.

Main features
-------------
- Total energy vs beta
- Kinetic energy vs beta
- Momentum vs beta
- Energy-momentum relation
- Classical vs relativistic kinetic energy
- Center-of-momentum velocity visualization for simple systems
- Compact dynamics summary figure

All public plotting functions return ``(fig, ax)`` except
``plot_dynamics_summary``, which returns ``(fig, axes)``.

Recommended location
--------------------
Save this file as:

    relativity/plotting/dynamics.py
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from relativity.constants import C
except Exception:  # pragma: no cover - fallback for standalone use
    C = 299_792_458.0

try:
    from .style import (
        set_default_style,
        get_axis,
        apply_axes_style,
        finalize_plot,
        require_matplotlib,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    import matplotlib.pyplot as _plt

    def require_matplotlib():
        return _plt

    def set_default_style(**kwargs):
        return None

    def get_axis(ax=None, *, figsize=None):
        if ax is not None:
            return ax.figure, ax
        return _plt.subplots(figsize=figsize)

    def apply_axes_style(
        ax,
        *,
        xlabel=None,
        ylabel=None,
        title=None,
        grid=True,
        equal=False,
        legend=False,
    ):
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

    def finalize_plot(fig, ax, *, show=False, tight_layout=True, return_fig_ax=True):
        if tight_layout:
            fig.tight_layout()
        if show:
            _plt.show()
        return (fig, ax) if return_fig_ax else ax

try:
    from .utils import beta_grid, finite_limits, as_numeric_array
except Exception:  # pragma: no cover - fallback for standalone use
    def beta_grid(beta_min=0.0, beta_max=0.999, n=500, *, include_endpoint=True):
        if beta_min <= -1 or beta_max >= 1:
            raise ValueError("beta values must satisfy -1 < beta < 1.")
        if n < 2:
            raise ValueError("n must be at least 2.")
        return np.linspace(beta_min, beta_max, n, endpoint=include_endpoint)

    def finite_limits(*arrays, margin=0.05):
        data = []
        for array in arrays:
            arr = np.asarray(array, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                data.append(arr)
        if not data:
            return -1.0, 1.0
        values = np.concatenate(data)
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            delta = 1.0 if lo == 0 else abs(lo) * margin
            return lo - delta, hi + delta
        delta = (hi - lo) * margin
        return lo - delta, hi + delta

    def as_numeric_array(data, substitutions=None):
        return np.asarray(data, dtype=float)


# -----------------------------------------------------------------------------
# Numeric formulas used for plotting
# -----------------------------------------------------------------------------


def _validate_mass(mass: float) -> float:
    mass = float(mass)
    if mass < 0:
        raise ValueError("mass must be non-negative.")
    return mass


def _validate_c(c: float) -> float:
    c = float(c)
    if c <= 0:
        raise ValueError("c must be positive.")
    return c


def _gamma_from_beta(beta) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    if np.any(np.abs(beta) >= 1):
        raise ValueError("beta must satisfy |beta| < 1.")
    return 1.0 / np.sqrt(1.0 - beta**2)


def _energy_from_beta(mass: float, beta, c: float) -> np.ndarray:
    return _gamma_from_beta(beta) * mass * c**2


def _kinetic_energy_from_beta(mass: float, beta, c: float) -> np.ndarray:
    return (_gamma_from_beta(beta) - 1.0) * mass * c**2


def _classical_kinetic_energy_from_beta(mass: float, beta, c: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    return 0.5 * mass * (beta * c) ** 2


def _momentum_from_beta(mass: float, beta, c: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    return _gamma_from_beta(beta) * mass * beta * c


def _energy_from_momentum(mass: float, p, c: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.sqrt((p * c) ** 2 + (mass * c**2) ** 2)


def _normalize_energy(values, mass: float, c: float, normalize: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not normalize:
        return values
    rest = mass * c**2
    if rest == 0:
        raise ValueError("Cannot normalize by rest energy when mass is zero.")
    return values / rest


def _normalize_momentum(values, mass: float, c: float, normalize: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not normalize:
        return values
    scale = mass * c
    if scale == 0:
        raise ValueError("Cannot normalize by m c when mass is zero.")
    return values / scale


def _setup_axis(ax=None, figsize: Tuple[float, float] = (7.5, 5.0)):
    set_default_style()
    return get_axis(ax=ax, figsize=figsize)


def _finish(
    fig,
    ax,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    legend: bool = True,
    show: bool = False,
    tight_layout: bool = True,
):
    apply_axes_style(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        grid=True,
        legend=legend,
    )
    return finalize_plot(fig, ax, show=show, tight_layout=tight_layout)


# -----------------------------------------------------------------------------
# Public plotting functions
# -----------------------------------------------------------------------------


def plot_total_energy_vs_beta(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    n: int = 500,
    normalize: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot total relativistic energy ``E = gamma m c^2`` vs ``beta``.

    If ``normalize=True``, the vertical axis is ``E / (m c^2) = gamma``.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    beta = beta_grid(beta_min, beta_max, n)
    energy = _energy_from_beta(mass, beta, c)
    y = _normalize_energy(energy, mass, c, normalize)

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    label = r"$E/(mc^2)=\gamma$" if normalize else r"$E=\gamma mc^2$"
    ax.plot(beta, y, label=label)

    ylabel = r"$E/(mc^2)$" if normalize else "Energy E"
    return _finish(
        fig,
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=ylabel,
        title="Total Relativistic Energy vs Beta",
        show=show,
    )


def plot_kinetic_energy_vs_beta(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    n: int = 500,
    normalize: bool = True,
    compare_classical: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot relativistic kinetic energy vs beta.

    If ``compare_classical=True``, also plots ``K = 1/2 m v^2``.
    If ``normalize=True``, energies are divided by ``m c^2``.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    beta = beta_grid(beta_min, beta_max, n)

    k_rel = _kinetic_energy_from_beta(mass, beta, c)
    y_rel = _normalize_energy(k_rel, mass, c, normalize)

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(beta, y_rel, label=r"Relativistic $K=(\gamma-1)mc^2$")

    if compare_classical:
        k_classical = _classical_kinetic_energy_from_beta(mass, beta, c)
        y_classical = _normalize_energy(k_classical, mass, c, normalize)
        ax.plot(beta, y_classical, linestyle="--", alpha=0.85, label=r"Classical $K=\frac{1}{2}mv^2$")

    ylabel = r"$K/(mc^2)$" if normalize else "Kinetic energy K"
    return _finish(
        fig,
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=ylabel,
        title="Kinetic Energy vs Beta",
        show=show,
    )


def plot_momentum_vs_beta(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    n: int = 500,
    normalize: bool = True,
    compare_classical: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot 1D relativistic momentum ``p = gamma m v`` vs beta.

    If ``normalize=True``, the vertical axis is ``p/(mc)``.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    beta = beta_grid(beta_min, beta_max, n)

    p_rel = _momentum_from_beta(mass, beta, c)
    y_rel = _normalize_momentum(p_rel, mass, c, normalize)

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(beta, y_rel, label=r"Relativistic $p=\gamma mv$")

    if compare_classical:
        p_classical = mass * beta * c
        y_classical = _normalize_momentum(p_classical, mass, c, normalize)
        ax.plot(beta, y_classical, linestyle="--", alpha=0.85, label=r"Classical $p=mv$")

    ax.axhline(0.0, linestyle=":", alpha=0.6)
    ax.axvline(0.0, linestyle=":", alpha=0.6)

    ylabel = r"$p/(mc)$" if normalize else "Momentum p"
    return _finish(
        fig,
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=ylabel,
        title="Relativistic Momentum vs Beta",
        show=show,
    )


def plot_energy_momentum_relation(
    mass: float = 1.0,
    *,
    c: float = C,
    p_min: float = -5.0,
    p_max: float = 5.0,
    n: int = 500,
    normalized: bool = True,
    include_negative_energy: bool = False,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot the mass-shell relation ``E^2 = p^2 c^2 + m^2 c^4``.

    With ``normalized=True``, horizontal values are interpreted as ``p/(mc)``
    and the vertical axis is ``E/(mc^2)``.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    if mass == 0 and normalized:
        raise ValueError("normalized=True requires nonzero mass.")
    if p_min >= p_max:
        raise ValueError("p_min must be smaller than p_max.")
    if n < 2:
        raise ValueError("n must be at least 2.")

    p_axis = np.linspace(p_min, p_max, n)
    p_physical = p_axis * mass * c if normalized else p_axis
    energy = _energy_from_momentum(mass, p_physical, c)
    y = _normalize_energy(energy, mass, c, normalized)

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(p_axis, y, label=r"$E^2=p^2c^2+m^2c^4$")

    if include_negative_energy:
        ax.plot(p_axis, -y, linestyle="--", alpha=0.7, label="Negative-energy branch")

    ax.axhline(0.0, linestyle=":", alpha=0.6)
    ax.axvline(0.0, linestyle=":", alpha=0.6)

    xlabel = r"$p/(mc)$" if normalized else "Momentum p"
    ylabel = r"$E/(mc^2)$" if normalized else "Energy E"
    return _finish(
        fig,
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title="Energy-Momentum Relation",
        show=show,
    )


def plot_energy_components_vs_beta(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    n: int = 500,
    normalize: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot rest energy, kinetic energy and total energy vs beta."""
    mass = _validate_mass(mass)
    c = _validate_c(c)
    beta = beta_grid(beta_min, beta_max, n)

    rest = np.full_like(beta, mass * c**2, dtype=float)
    kinetic = _kinetic_energy_from_beta(mass, beta, c)
    total = rest + kinetic

    rest_y = _normalize_energy(rest, mass, c, normalize)
    kinetic_y = _normalize_energy(kinetic, mass, c, normalize)
    total_y = _normalize_energy(total, mass, c, normalize)

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(beta, rest_y, linestyle="--", label=r"Rest energy $mc^2$")
    ax.plot(beta, kinetic_y, label=r"Kinetic energy $K$")
    ax.plot(beta, total_y, label=r"Total energy $E$")

    ylabel = r"Energy / $(mc^2)$" if normalize else "Energy"
    return _finish(
        fig,
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=ylabel,
        title="Relativistic Energy Components",
        show=show,
    )


def plot_classical_error_vs_beta(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_min: float = 0.01,
    beta_max: float = 0.99,
    n: int = 500,
    percent: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot the relative error of classical kinetic energy.

    Error is defined as ``(K_classical - K_relativistic) / K_relativistic``.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    beta = beta_grid(beta_min, beta_max, n)

    k_rel = _kinetic_energy_from_beta(mass, beta, c)
    k_classical = _classical_kinetic_energy_from_beta(mass, beta, c)
    error = (k_classical - k_rel) / k_rel
    y = 100.0 * error if percent else error

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(beta, y, label="Classical kinetic-energy error")
    ax.axhline(0.0, linestyle=":", alpha=0.6)

    ylabel = "Relative error (%)" if percent else "Relative error"
    return _finish(
        fig,
        ax,
        xlabel=r"$\beta = v/c$",
        ylabel=ylabel,
        title="Classical Approximation Error",
        show=show,
    )


def plot_com_velocity_1d(
    masses: Sequence[float],
    betas: Sequence[float],
    *,
    c: float = C,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Visualize a 1D system and mark its center-of-momentum velocity.

    For particles moving along x,

        beta_COM = (sum p_i c) / (sum E_i)

    The plot shows each particle beta on a horizontal axis and marks the COM
    beta as a vertical line.
    """
    c = _validate_c(c)
    masses = np.asarray(masses, dtype=float)
    betas = np.asarray(betas, dtype=float)

    if masses.ndim != 1 or betas.ndim != 1 or masses.size != betas.size:
        raise ValueError("masses and betas must be one-dimensional arrays of the same length.")
    if masses.size == 0:
        raise ValueError("At least one particle is required.")
    if np.any(masses < 0):
        raise ValueError("masses must be non-negative.")
    if np.any(np.abs(betas) >= 1):
        raise ValueError("Every beta must satisfy |beta| < 1.")

    gammas = _gamma_from_beta(betas)
    energies = gammas * masses * c**2
    momenta = gammas * masses * betas * c
    beta_com = float(np.sum(momenta) * c / np.sum(energies))

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    y = np.zeros_like(betas)
    sizes = 80.0 + 120.0 * masses / masses.max() if masses.max() > 0 else 100.0
    ax.scatter(betas, y, s=sizes, label="Particles")

    for i, b in enumerate(betas):
        ax.annotate(f"{i}", xy=(b, 0), xytext=(0, 10), textcoords="offset points", ha="center")

    ax.axvline(beta_com, linestyle="--", label=fr"$\beta_{{COM}}={beta_com:.4g}$")
    ax.axhline(0.0, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.5, 0.8)
    ax.set_xlim(*finite_limits(betas, [beta_com], margin=0.15))
    ax.set_yticks([])

    return _finish(
        fig,
        ax,
        xlabel=r"$\beta_x = v_x/c$",
        ylabel="",
        title="1D Center-of-Momentum Velocity",
        show=show,
    )


def plot_force_power_constant_force_1d(
    force: float = 1.0,
    mass: float = 1.0,
    *,
    c: float = C,
    t_min: float = 0.0,
    t_max: float = 10.0,
    n: int = 500,
    normalize_time: bool = True,
    ax=None,
    show: bool = False,
    figsize: Tuple[float, float] = (7.5, 5.0),
):
    """Plot beta(t) and normalized power for constant 1D force.

    For a particle initially at rest under constant lab-frame force ``F``:

        p(t) = F t,
        beta(t) = p / sqrt((mc)^2 + p^2),
        P(t) = F v.

    If ``normalize_time=True``, the horizontal variable is
    ``theta = F t / (m c)`` and ``t_min``/``t_max`` are interpreted as theta.
    Otherwise ``t_min``/``t_max`` are physical times in seconds.
    """
    mass = _validate_mass(mass)
    c = _validate_c(c)
    force = float(force)
    if mass == 0:
        raise ValueError("mass must be nonzero for constant-force motion.")
    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max.")
    if n < 2:
        raise ValueError("n must be at least 2.")

    x = np.linspace(t_min, t_max, n)
    theta = x if normalize_time else force * x / (mass * c)
    beta = theta / np.sqrt(1.0 + theta**2)
    power_norm = beta  # P/(F c) = beta

    fig, ax = _setup_axis(ax=ax, figsize=figsize)
    ax.plot(x, beta, label=r"$\beta(t)$")
    ax.plot(x, power_norm, linestyle="--", alpha=0.85, label=r"$P/(Fc)$")
    ax.axhline(1.0, linestyle=":", alpha=0.6)

    xlabel = r"$Ft/(mc)$" if normalize_time else "Time t"
    return _finish(
        fig,
        ax,
        xlabel=xlabel,
        ylabel="Dimensionless value",
        title="Constant Force Motion in 1D",
        show=show,
    )


def plot_dynamics_summary(
    mass: float = 1.0,
    *,
    c: float = C,
    beta_max: float = 0.95,
    n: int = 400,
    normalize: bool = True,
    show: bool = False,
    figsize: Tuple[float, float] = (11.0, 8.0),
):
    """Create a compact 2x2 summary of relativistic dynamics plots."""
    set_default_style()
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_total_energy_vs_beta(
        mass,
        c=c,
        beta_min=0.0,
        beta_max=beta_max,
        n=n,
        normalize=normalize,
        ax=axes[0, 0],
    )
    plot_kinetic_energy_vs_beta(
        mass,
        c=c,
        beta_min=0.0,
        beta_max=beta_max,
        n=n,
        normalize=normalize,
        compare_classical=True,
        ax=axes[0, 1],
    )
    plot_momentum_vs_beta(
        mass,
        c=c,
        beta_min=-beta_max,
        beta_max=beta_max,
        n=n,
        normalize=normalize,
        compare_classical=True,
        ax=axes[1, 0],
    )
    plot_energy_momentum_relation(
        mass,
        c=c,
        p_min=-5.0,
        p_max=5.0,
        n=n,
        normalized=normalize,
        ax=axes[1, 1],
    )

    fig.suptitle("Special Relativity Dynamics Summary")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


__all__ = [
    "plot_total_energy_vs_beta",
    "plot_kinetic_energy_vs_beta",
    "plot_momentum_vs_beta",
    "plot_energy_momentum_relation",
    "plot_energy_components_vs_beta",
    "plot_classical_error_vs_beta",
    "plot_com_velocity_1d",
    "plot_force_power_constant_force_1d",
    "plot_dynamics_summary",
]
