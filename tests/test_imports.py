"""Smoke tests: modules should import without side effects."""
from __future__ import annotations

import importlib


MODULES = [
    "relativity",
    "relativity.constants",
    "relativity.utils",
    "relativity.math.minkowski",
    "relativity.math.lorentz",
    "relativity.math.tensors",
    "relativity.physics.fourvector",
    "relativity.physics.event",
    "relativity.physics.frame",
    "relativity.physics.particle",
    "relativity.physics.photon",
    "relativity.physics.radiation",
    "relativity.physics.worldline",
    "relativity.sr.kinematics",
    "relativity.sr.dynamics",
    "relativity.sr.doppler",
    "relativity.sr.decay",
    "relativity.sr.collision",
]


def test_core_modules_import():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_plotting_modules_import():
    plotting_modules = [
        "relativity.plotting.style",
        "relativity.plotting.utils",
        "relativity.plotting.spacetime",
        "relativity.plotting.worldline",
        "relativity.plotting.kinematics",
        "relativity.plotting.dynamics",
        "relativity.plotting.decay",
    ]
    for module_name in plotting_modules:
        importlib.import_module(module_name)
