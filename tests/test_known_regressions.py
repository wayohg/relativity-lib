"""
Regression tests for bugs already identified during review.

These are marked xfail because the current code may still contain these bugs.
When you fix each bug, remove its xfail mark so it becomes a normal test.
"""
from __future__ import annotations

import numpy as np
import pytest

from relativity.physics.frame import ReferenceFrame
from relativity.physics.photon import Photon
from relativity.physics.particle import Particle
from relativity.physics.event import Event
from relativity.physics.worldline import Worldline
from relativity.sr.decay import decay_length, lab_time_from_distance


def test_photon_rejects_zero_direction():
    with pytest.raises(ValueError):
        Photon(frequency=1.0, direction=[0, 0, 0], c=1.0, h=1.0)


def test_particle_from_energy_rejects_zero_direction():
    with pytest.raises(ValueError):
        Particle.from_energy(mass=1.0, energy=2.0, direction=[0, 0, 0], c=1.0)


def test_decay_rejects_negative_scalar_speed():
    with pytest.raises(ValueError):
        decay_length(1.0, -0.5, c=1.0)
    with pytest.raises(ValueError):
        lab_time_from_distance(10.0, -0.5, c=1.0)


def test_worldline_constructor_sorts_numeric_events():
    wl = Worldline([Event(2, [2, 0, 0], c=1), Event(0, [0, 0, 0], c=1)], c=1)
    assert np.allclose(wl.times, [0, 2])

def test_worldline_proper_time_rejects_spacelike_segment():
    wl = Worldline([Event(0, [0, 0, 0], c=1), Event(1, [2, 0, 0], c=1)], c=1)
    with pytest.raises(ValueError):
        wl.proper_time()


def test_reference_frame_velocity_wrt_sign_convention():
    S = ReferenceFrame("S", [0, 0, 0], c=1)
    Sp = ReferenceFrame("S'", [0.5, 0, 0], relative_to=S, c=1)
    assert np.allclose(S.velocity_wrt(Sp), [-0.5, 0, 0])
