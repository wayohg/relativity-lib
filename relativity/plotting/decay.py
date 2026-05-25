"""Decay plotting tools for special relativity.

This module contains numeric visualization helpers for the decay utilities in
``relativity.sr.decay``.  It does not introduce new physics; it only samples
existing decay formulas over convenient numeric grids and draws them with
Matplotlib.

Typical use
-----------
>>> from relativity.plotting.decay import plot_survival_probability_time
>>> plot_survival_probability_time(proper_lifetime=2.2e-6, velocity=0.99, c=1, show=True)

All public functions return ``(fig, ax)`` except ``plot_decay_summary``, which
returns ``(fig, axes)``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from relativity.plotting.style import apply_axes_style, finalize_plot, get_axis
from relativity.plotting.utils import finite_limits

try:
    from relativity.constants import C as DEFAULT_C
except Exception:  # pragma: no cover - defensive fallback
    DEFAULT_C = 299_792_458.0

try:
    from relativity.sr import decay as sr_decay
except Exception:  # pragma: no cover - allows isolated use during development
    sr_decay = None


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _beta_grid(beta_min: float = 0.0, beta_max: float = 0.99, n: int = 500) -> np.ndarray:
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_min < 0:
        raise ValueError("beta_min must be nonnegative for decay plots.")
    if beta_max >= 1:
        raise ValueError("beta_max must be strictly less than 1.")
    if beta_min >= beta_max:
        raise ValueError("beta_min must be smaller than beta_max.")
    if n < 2:
        raise ValueError("n must be at least 2.")
    return np.linspace(beta_min, beta_max, int(n))


def _gamma(beta: np.ndarray | float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    return 1.0 / np.sqrt(1.0 - beta**2)


def _speed_value(velocity, c: float = DEFAULT_C) -> float:
    """Accept scalar speed, beta-like scalar, or a 3-vector velocity."""
    c = float(c)
    arr = np.asarray(velocity, dtype=float)
    if arr.shape == ():
        speed = abs(float(arr))
        # Convenience for natural-unit plotting: when c != 1 this treats
        # values in [0, 1) as beta fractions.  If the user wants an actual
        # speed below 1 m/s, they can pass c=1 or use a 3-vector.
        if c != 1.0 and 0 <= speed < 1:
            return speed * c
        return speed
    if arr.shape == (3,):
        return float(np.linalg.norm(arr))
    raise ValueError("velocity must be a scalar speed/beta or a 3-vector.")


def _validate_speed(speed: float, c: float) -> float:
    speed = abs(float(speed))
    c = _positive(c, "c")
    if speed >= c:
        raise ValueError("massive-particle speed must satisfy |v| < c.")
    return speed


def _lab_lifetime(proper_lifetime: float | None, velocity=None, *, lab_lifetime_value=None, c=DEFAULT_C) -> float:
    if lab_lifetime_value is not None:
        return _positive(lab_lifetime_value, "lab_lifetime_value")
    if proper_lifetime is None or velocity is None:
        raise ValueError("Provide either lab_lifetime_value or both proper_lifetime and velocity.")
    proper_lifetime = _positive(proper_lifetime, "proper_lifetime")
    speed = _validate_speed(_speed_value(velocity, c), c)
    if sr_decay is not None:
        return float(sr_decay.lab_lifetime(proper_lifetime, speed, c=c))
    beta = speed / float(c)
    return float(_gamma(beta) * proper_lifetime)


def _decay_length(proper_lifetime: float, velocity, c=DEFAULT_C) -> float:
    proper_lifetime = _positive(proper_lifetime, "proper_lifetime")
    speed = _validate_speed(_speed_value(velocity, c), c)
    if sr_decay is not None:
        return float(sr_decay.decay_length(proper_lifetime, speed, c=c))
    return speed * _lab_lifetime(proper_lifetime, speed, c=c)


def _time_grid(time_max: float | None, tau_lab: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2.")
    if time_max is None:
        time_max = 5.0 * tau_lab
    time_max = _positive(time_max, "time_max")
    return np.linspace(0.0, time_max, int(n))


def _distance_grid(distance_max: float | None, mean_length: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2.")
    if distance_max is None:
        distance_max = 5.0 * mean_length
    distance_max = _positive(distance_max, "distance_max")
    return np.linspace(0.0, distance_max, int(n))


def _survival_time(t: np.ndarray, tau_lab: float) -> np.ndarray:
    return np.exp(-np.asarray(t, dtype=float) / tau_lab)


def _survival_distance(x: np.ndarray, mean_length: float) -> np.ndarray:
    return np.exp(-np.asarray(x, dtype=float) / mean_length)


def _decay_probability_from_survival(survival: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(survival, dtype=float)


def _as_float_pair(pair, name: str = "pair") -> tuple[float, float]:
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return float(pair[0]), float(pair[1])


def _two_body_summary(parent_mass, daughter_mass_1, daughter_mass_2, c=DEFAULT_C) -> dict:
    if sr_decay is not None:
        summary = sr_decay.two_body_decay_summary(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
        return {
            "Q": float(summary["Q"]),
            "momentum_magnitude": float(summary["momentum_magnitude"]),
            "energies": _as_float_pair(summary["energies"], "energies"),
            "kinetic_energies": _as_float_pair(summary["kinetic_energies"], "kinetic_energies"),
            "betas": _as_float_pair(summary["betas"], "betas"),
        }

    M = _positive(parent_mass, "parent_mass")
    m1 = _nonnegative(daughter_mass_1, "daughter_mass_1")
    m2 = _nonnegative(daughter_mass_2, "daughter_mass_2")
    c = _positive(c, "c")
    if M < m1 + m2:
        raise ValueError("Two-body decay is not energetically allowed: M >= m1 + m2 is required.")

    radicand = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    p = c * math.sqrt(max(radicand, 0.0)) / (2.0 * M)
    E1 = (M**2 + m1**2 - m2**2) * c**2 / (2.0 * M)
    E2 = (M**2 + m2**2 - m1**2) * c**2 / (2.0 * M)
    K1 = E1 - m1 * c**2
    K2 = E2 - m2 * c**2
    b1 = 1.0 if m1 == 0 else p * c / E1
    b2 = 1.0 if m2 == 0 else p * c / E2
    return {
        "Q": (M - m1 - m2) * c**2,
        "momentum_magnitude": p,
        "energies": (E1, E2),
        "kinetic_energies": (K1, K2),
        "betas": (b1, b2),
    }


def _bar_labels(ax, values: Sequence[float], fmt: str = "{:.3g}") -> None:
    """Add compact numeric labels over bars."""
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = 0.02 * y_span if y_span > 0 else 0.02
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_survival_probability_time(
    *,
    proper_lifetime: float | None = None,
    velocity=None,
    lab_lifetime_value: float | None = None,
    time_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    label: str | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot survival probability ``P(t)=exp(-t/tau_lab)`` versus lab time."""
    tau_lab = _lab_lifetime(proper_lifetime, velocity, lab_lifetime_value=lab_lifetime_value, c=c)
    t = _time_grid(time_max, tau_lab, n)
    P = _survival_time(t, tau_lab)

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(t, P, label=label or r"$P(t)=e^{-t/\tau_{lab}}$", **kwargs)
    ax.axhline(math.exp(-1), linestyle="--", alpha=0.7, label=r"$1/e$")
    ax.axvline(tau_lab, linestyle="--", alpha=0.7, label=r"$\tau_{lab}$")
    apply_axes_style(
        ax,
        xlabel=r"Lab time $t$",
        ylabel="Survival probability",
        title="Survival probability vs lab time",
        legend=True,
    )
    ax.set_ylim(-0.03, 1.03)
    return finalize_plot(fig, ax, show=show)


def plot_decay_probability_time(
    *,
    proper_lifetime: float | None = None,
    velocity=None,
    lab_lifetime_value: float | None = None,
    time_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    label: str | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot decay probability ``1 - exp(-t/tau_lab)`` versus lab time."""
    tau_lab = _lab_lifetime(proper_lifetime, velocity, lab_lifetime_value=lab_lifetime_value, c=c)
    t = _time_grid(time_max, tau_lab, n)
    P_decay = _decay_probability_from_survival(_survival_time(t, tau_lab))

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(t, P_decay, label=label or r"$1-e^{-t/\tau_{lab}}$", **kwargs)
    ax.axvline(tau_lab, linestyle="--", alpha=0.7, label=r"$\tau_{lab}$")
    apply_axes_style(
        ax,
        xlabel=r"Lab time $t$",
        ylabel="Decay probability",
        title="Decay probability vs lab time",
        legend=True,
    )
    ax.set_ylim(-0.03, 1.03)
    return finalize_plot(fig, ax, show=show)


def plot_remaining_count(
    *,
    initial_count: float = 1.0,
    proper_lifetime: float | None = None,
    velocity=None,
    lab_lifetime_value: float | None = None,
    time_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    show_decayed: bool = True,
    show: bool = False,
    **line_kwargs,
):
    """Plot expected remaining particles ``N(t)=N0 exp(-t/tau_lab)``."""
    initial_count = _nonnegative(initial_count, "initial_count")
    tau_lab = _lab_lifetime(proper_lifetime, velocity, lab_lifetime_value=lab_lifetime_value, c=c)
    t = _time_grid(time_max, tau_lab, n)
    N = initial_count * _survival_time(t, tau_lab)

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(t, N, label="Remaining", **kwargs)
    if show_decayed:
        ax.plot(t, initial_count - N, linestyle="--", label="Decayed")
    ax.axvline(tau_lab, linestyle=":", alpha=0.7, label=r"$\tau_{lab}$")
    apply_axes_style(
        ax,
        xlabel=r"Lab time $t$",
        ylabel="Expected count",
        title="Exponential decay counts",
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)


def plot_activity_time(
    *,
    initial_count: float = 1.0,
    proper_lifetime: float | None = None,
    velocity=None,
    lab_lifetime_value: float | None = None,
    time_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    show: bool = False,
    **line_kwargs,
):
    """Plot activity/rate ``A(t)=N(t)/tau_lab`` versus lab time."""
    initial_count = _nonnegative(initial_count, "initial_count")
    tau_lab = _lab_lifetime(proper_lifetime, velocity, lab_lifetime_value=lab_lifetime_value, c=c)
    t = _time_grid(time_max, tau_lab, n)
    A = initial_count * _survival_time(t, tau_lab) / tau_lab

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(t, A, label=r"$A(t)=N(t)/\tau_{lab}$", **kwargs)
    apply_axes_style(
        ax,
        xlabel=r"Lab time $t$",
        ylabel="Activity",
        title="Activity vs lab time",
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)


def plot_survival_probability_distance(
    *,
    proper_lifetime: float,
    velocity,
    distance_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    label: str | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot survival probability after traveling distance ``x`` in the lab frame."""
    L = _decay_length(proper_lifetime, velocity, c=c)
    x = _distance_grid(distance_max, L, n)
    P = _survival_distance(x, L)

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(x, P, label=label or r"$P(x)=e^{-x/L}$", **kwargs)
    ax.axhline(math.exp(-1), linestyle="--", alpha=0.7, label=r"$1/e$")
    ax.axvline(L, linestyle="--", alpha=0.7, label="mean decay length")
    apply_axes_style(
        ax,
        xlabel="Lab distance",
        ylabel="Survival probability",
        title="Survival probability vs distance",
        legend=True,
    )
    ax.set_ylim(-0.03, 1.03)
    return finalize_plot(fig, ax, show=show)


def plot_decay_probability_distance(
    *,
    proper_lifetime: float,
    velocity,
    distance_max: float | None = None,
    n: int = 500,
    c: float = DEFAULT_C,
    ax=None,
    label: str | None = None,
    show: bool = False,
    **line_kwargs,
):
    """Plot probability of decaying before distance ``x``."""
    L = _decay_length(proper_lifetime, velocity, c=c)
    x = _distance_grid(distance_max, L, n)
    P_decay = _decay_probability_from_survival(_survival_distance(x, L))

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(x, P_decay, label=label or r"$1-e^{-x/L}$", **kwargs)
    ax.axvline(L, linestyle="--", alpha=0.7, label="mean decay length")
    apply_axes_style(
        ax,
        xlabel="Lab distance",
        ylabel="Decay probability",
        title="Decay probability vs distance",
        legend=True,
    )
    ax.set_ylim(-0.03, 1.03)
    return finalize_plot(fig, ax, show=show)


def plot_decay_length_vs_beta(
    *,
    proper_lifetime: float,
    c: float = DEFAULT_C,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    n: int = 500,
    ax=None,
    show: bool = False,
    **line_kwargs,
):
    """Plot mean decay length ``L = beta gamma c tau0`` versus beta."""
    proper_lifetime = _positive(proper_lifetime, "proper_lifetime")
    c = _positive(c, "c")
    beta = _beta_grid(beta_min, beta_max, n)
    L = beta * _gamma(beta) * c * proper_lifetime

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(beta, L, label=r"$L=\beta\gamma c\tau_0$", **kwargs)
    apply_axes_style(
        ax,
        xlabel=r"$\beta=v/c$",
        ylabel="Mean decay length",
        title="Mean decay length vs beta",
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)


def plot_lab_lifetime_vs_beta(
    *,
    proper_lifetime: float,
    beta_min: float = 0.0,
    beta_max: float = 0.99,
    n: int = 500,
    ax=None,
    show: bool = False,
    **line_kwargs,
):
    """Plot dilated lifetime ``tau_lab = gamma tau0`` versus beta."""
    proper_lifetime = _positive(proper_lifetime, "proper_lifetime")
    beta = _beta_grid(beta_min, beta_max, n)
    tau_lab = _gamma(beta) * proper_lifetime

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(beta, tau_lab, label=r"$\tau_{lab}=\gamma\tau_0$", **kwargs)
    ax.axhline(proper_lifetime, linestyle="--", alpha=0.7, label=r"$\tau_0$")
    apply_axes_style(
        ax,
        xlabel=r"$\beta=v/c$",
        ylabel=r"Lab lifetime $\tau_{lab}$",
        title="Dilated lifetime vs beta",
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)


def plot_required_speed_for_survival(
    *,
    proper_lifetime: float,
    survival_fraction: float,
    distance_max: float,
    c: float = DEFAULT_C,
    n: int = 500,
    ax=None,
    show: bool = False,
    **line_kwargs,
):
    """Plot required beta so a fixed survival fraction reaches each distance.

    For ``f = exp[-x/(beta gamma c tau0)]`` this inverts the relation to
    ``beta = bg/sqrt(1+bg^2)``, where ``bg=x/[-c tau0 ln(f)]``.
    """
    proper_lifetime = _positive(proper_lifetime, "proper_lifetime")
    c = _positive(c, "c")
    f = float(survival_fraction)
    if not (0 < f < 1):
        raise ValueError("survival_fraction must satisfy 0 < f < 1.")
    distance_max = _positive(distance_max, "distance_max")
    if n < 2:
        raise ValueError("n must be at least 2.")

    x = np.linspace(0.0, distance_max, int(n))
    bg = x / (-c * proper_lifetime * math.log(f))
    beta = bg / np.sqrt(1.0 + bg**2)

    fig, ax = get_axis(ax)
    kwargs = {"linewidth": 1.8}
    kwargs.update(line_kwargs)
    ax.plot(x, beta, label=fr"Required $\beta$ for survival fraction {f:g}", **kwargs)
    ax.set_ylim(-0.03, 1.03)
    apply_axes_style(
        ax,
        xlabel="Lab distance",
        ylabel=r"Required $\beta$",
        title="Required speed for a survival fraction",
        legend=True,
    )
    return finalize_plot(fig, ax, show=show)


def plot_two_body_decay_bars(
    *,
    parent_mass: float,
    daughter_mass_1: float,
    daughter_mass_2: float,
    c: float = DEFAULT_C,
    labels: tuple[str, str] = ("daughter 1", "daughter 2"),
    quantity: str = "kinetic_energy",
    ax=None,
    show_values: bool = True,
    show: bool = False,
    **bar_kwargs,
):
    """Bar plot for a two-body decay in the parent rest frame.

    Parameters
    ----------
    quantity:
        One of ``"energy"``, ``"kinetic_energy"`` or ``"beta"``.
    """
    summary = _two_body_summary(parent_mass, daughter_mass_1, daughter_mass_2, c=c)
    quantity_key = quantity.strip().lower().replace("-", "_")

    if quantity_key in {"energy", "energies", "total_energy"}:
        values = summary["energies"]
        ylabel = "Total energy"
        title = "Two-body decay energies"
    elif quantity_key in {"kinetic", "kinetic_energy", "kinetic_energies"}:
        values = summary["kinetic_energies"]
        ylabel = "Kinetic energy"
        title = "Two-body decay kinetic energies"
    elif quantity_key in {"beta", "betas", "speed_fraction"}:
        values = summary["betas"]
        ylabel = r"$\beta=v/c$"
        title = "Two-body decay daughter betas"
    else:
        raise ValueError("quantity must be 'energy', 'kinetic_energy' or 'beta'.")

    fig, ax = get_axis(ax)
    kwargs = {}
    kwargs.update(bar_kwargs)
    ax.bar(list(labels), list(values), **kwargs)
    if quantity_key in {"beta", "betas", "speed_fraction"}:
        ax.set_ylim(0.0, max(1.05, 1.08 * max(values)))
    else:
        lo, hi = finite_limits(values, margin=0.12)
        ax.set_ylim(0.0, max(hi, 1e-12))
    if show_values:
        _bar_labels(ax, values)
    apply_axes_style(ax, ylabel=ylabel, title=title)
    return finalize_plot(fig, ax, show=show)


def plot_two_body_decay_summary(
    *,
    parent_mass: float,
    daughter_mass_1: float,
    daughter_mass_2: float,
    c: float = DEFAULT_C,
    labels: tuple[str, str] = ("daughter 1", "daughter 2"),
    figsize: tuple[float, float] = (11.0, 4.5),
    show: bool = False,
):
    """Create a compact summary of daughter energies, kinetic energies and betas."""
    from relativity.plotting.style import require_matplotlib

    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_two_body_decay_bars(
        parent_mass=parent_mass,
        daughter_mass_1=daughter_mass_1,
        daughter_mass_2=daughter_mass_2,
        c=c,
        labels=labels,
        quantity="energy",
        ax=axes[0],
        show=False,
    )
    plot_two_body_decay_bars(
        parent_mass=parent_mass,
        daughter_mass_1=daughter_mass_1,
        daughter_mass_2=daughter_mass_2,
        c=c,
        labels=labels,
        quantity="kinetic_energy",
        ax=axes[1],
        show=False,
    )
    plot_two_body_decay_bars(
        parent_mass=parent_mass,
        daughter_mass_1=daughter_mass_1,
        daughter_mass_2=daughter_mass_2,
        c=c,
        labels=labels,
        quantity="beta",
        ax=axes[2],
        show=False,
    )
    fig.suptitle("Two-body decay summary")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_decay_summary(
    *,
    proper_lifetime: float,
    velocity,
    initial_count: float = 1.0,
    c: float = DEFAULT_C,
    beta_max: float = 0.99,
    n: int = 500,
    figsize: tuple[float, float] = (11.0, 8.0),
    show: bool = False,
):
    """Create a 2x2 summary figure for relativistic decay.

    Panels:
    - survival probability vs lab time
    - remaining/decayed counts vs lab time
    - survival probability vs lab distance
    - mean decay length vs beta
    """
    from relativity.plotting.style import require_matplotlib

    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_survival_probability_time(
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        c=c,
        n=n,
        ax=axes[0, 0],
        show=False,
    )
    plot_remaining_count(
        initial_count=initial_count,
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        c=c,
        n=n,
        ax=axes[0, 1],
        show=False,
    )
    plot_survival_probability_distance(
        proper_lifetime=proper_lifetime,
        velocity=velocity,
        c=c,
        n=n,
        ax=axes[1, 0],
        show=False,
    )
    plot_decay_length_vs_beta(
        proper_lifetime=proper_lifetime,
        c=c,
        beta_max=beta_max,
        n=n,
        ax=axes[1, 1],
        show=False,
    )

    fig.suptitle("Relativistic decay summary")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


__all__ = [
    "plot_survival_probability_time",
    "plot_decay_probability_time",
    "plot_remaining_count",
    "plot_activity_time",
    "plot_survival_probability_distance",
    "plot_decay_probability_distance",
    "plot_decay_length_vs_beta",
    "plot_lab_lifetime_vs_beta",
    "plot_required_speed_for_survival",
    "plot_two_body_decay_bars",
    "plot_two_body_decay_summary",
    "plot_decay_summary",
]
