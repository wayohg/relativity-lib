"""Smoke tests for plotting functions. These do not verify visual appearance."""
from __future__ import annotations

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from relativity.physics.event import Event
from relativity.physics.worldline import Worldline
from relativity.plotting.spacetime import plot_light_cone, plot_event, plot_spacetime_diagram
from relativity.plotting.worldline import plot_worldline, plot_position_time
from relativity.plotting.kinematics import plot_gamma_vs_beta, plot_kinematics_summary
from relativity.plotting.dynamics import plot_total_energy_vs_beta, plot_dynamics_summary
from relativity.plotting.decay import plot_survival_probability_time, plot_decay_summary


def _close(fig):
    plt.close(fig)


def test_plot_spacetime_basic():
    fig, ax = plot_light_cone(c=1, show=False)
    plot_event(Event(1, [0.5, 0, 0], c=1), ax=ax, show=False)
    assert fig is ax.figure
    _close(fig)


def test_plot_spacetime_diagram_basic():
    events = [Event(0, [0, 0, 0], c=1), Event(1, [0.5, 0, 0], c=1)]
    wl = Worldline(events, c=1)
    fig, ax = plot_spacetime_diagram(events=events, worldlines=[wl], c=1, show=False)
    assert fig is ax.figure
    _close(fig)


def test_plot_worldline_and_position_time():
    wl = Worldline([Event(0, [0, 0, 0], c=1), Event(1, [0.5, 0, 0], c=1)], c=1)
    fig, ax = plot_worldline(wl, show=False)
    assert fig is ax.figure
    _close(fig)
    fig, ax = plot_position_time(wl, show=False)
    assert fig is ax.figure
    _close(fig)


def test_plot_kinematics_functions():
    fig, ax = plot_gamma_vs_beta(beta_max=0.9, show=False)
    assert fig is ax.figure
    _close(fig)
    fig, axes = plot_kinematics_summary(beta_max=0.9, show=False)
    assert axes.shape == (2, 2)
    _close(fig)


def test_plot_dynamics_functions():
    fig, ax = plot_total_energy_vs_beta(beta_max=0.9, show=False)
    assert fig is ax.figure
    _close(fig)
    fig, axes = plot_dynamics_summary(beta_max=0.9, show=False)
    assert axes.shape == (2, 2)
    _close(fig)


def test_plot_decay_functions():
    fig, ax = plot_survival_probability_time(proper_lifetime=1.0, velocity=0.5, c=1.0, show=False)
    assert fig is ax.figure
    _close(fig)
    fig, axes = plot_decay_summary(proper_lifetime=1.0, velocity=0.5, c=1.0, show=False)
    assert axes.shape == (2, 2)
    _close(fig)
