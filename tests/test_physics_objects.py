"""Tests for Event, FourVector, Particle, Photon, Worldline, and frames."""
from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from relativity.physics.event import Event
from relativity.physics.fourvector import FourVector
from relativity.physics.particle import Particle
from relativity.physics.photon import Photon
from relativity.physics.worldline import Worldline
from relativity.physics.frame import ReferenceFrame


def test_fourvector_minkowski_norm():
    X = FourVector(2, 1, 0, 0)
    assert X.interval_squared() == 3


def test_event_interval_to_other_event():
    A = Event(0, [0, 0, 0], c=1)
    B = Event(2, [1, 0, 0], c=1)
    assert B.proper_time_to(A) == pytest.approx(np.sqrt(3))


def test_particle_numeric_energy_momentum_invariant_mass():
    p = Particle(mass=2.0, velocity=[0.6, 0, 0], c=1.0)
    assert np.isclose(p.gamma, 1.25)
    assert np.isclose(p.energy, 2.5)
    assert np.allclose(p.momentum, [1.5, 0.0, 0.0])
    assert np.isclose(p.invariant_mass, 2.0)


def test_particle_symbolic_energy():
    m, v, c = sp.symbols("m v c", positive=True)
    p = Particle(mass=m, velocity=[v, 0, 0], c=c)
    expected = m*c**2 / sp.sqrt(1 - v**2/c**2)
    assert sp.simplify(p.energy - expected) == 0


def test_particle_rejects_light_speed_numeric():
    with pytest.raises(ValueError):
        Particle(mass=1.0, velocity=[1.0, 0, 0], c=1.0)


def test_photon_is_lightlike():
    ph = Photon(frequency=10.0, direction=[1, 0, 0], c=1.0, h=1.0)
    assert ph.is_lightlike()
    assert np.allclose(ph.velocity, [1, 0, 0])


def test_worldline_velocities_and_proper_time():
    wl = Worldline(
        [
            Event(0, [0, 0, 0], c=1),
            Event(2, [1, 0, 0], c=1),
        ],
        c=1,
    )
    vels = wl.velocities()
    assert np.allclose(vels[0], [0.5, 0, 0])
    assert np.isclose(wl.proper_time(), np.sqrt(3))


def test_reference_frame_four_velocity_at_rest():
    S = ReferenceFrame("S", [0, 0, 0], c=1)
    U = S.four_velocity()
    assert np.allclose(U.vec, [1, 0, 0, 0])
