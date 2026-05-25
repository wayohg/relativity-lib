"""Radiation and wave four-vector helpers."""
from __future__ import annotations
from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import minkowski_dot


class FourWaveVector(FourVector):
    """Wave four-vector k^mu=(omega/c,kx,ky,kz)."""
    def __init__(self, omega_over_c, k_vec):
        super().__init__(omega_over_c, *k_vec)

    def angular_frequency_measured_by(self, frame):
        u = frame.four_velocity()
        # With signature (+---), omega = k_mu u^mu for observer at rest gives omega.
        return minkowski_dot(self.vec, u.vec)

    def frequency_measured_by(self, frame):
        import numpy as np
        return self.angular_frequency_measured_by(frame) / (2 * np.pi)


def redshift(k, emitter_frame, observer_frame):
    omega_emit = k.angular_frequency_measured_by(emitter_frame)
    omega_obs = k.angular_frequency_measured_by(observer_frame)
    return omega_emit / omega_obs - 1
