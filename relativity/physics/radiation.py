"""Radiation and four-wave-vector helpers."""

from __future__ import annotations

from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import minkowski_dot
from relativity.utils import smart_array, smart_equal, is_symbolic, simplify


class FourWaveVector(FourVector):
    """Four-wave-vector k^mu=(omega/c, k_vec), usually null for light."""

    def __init__(self, omega_c, k_vec):
        k_vec = smart_array(k_vec)
        super().__init__(omega_c, *k_vec)

    def frequency_measured_by(self, frame):
        u = frame.four_velocity()
        return simplify(minkowski_dot(self.vec, u.vec))

    @staticmethod
    def redshift(k, emitter_frame, observer_frame):
        omega_emit = k.frequency_measured_by(emitter_frame)
        omega_obs = k.frequency_measured_by(observer_frame)
        if not is_symbolic(omega_obs) and omega_obs == 0:
            raise ValueError("Observer frequency is zero; cannot compute redshift.")
        return simplify(omega_emit / omega_obs - 1)

    def is_lightlike(self, tol=1e-10):
        return smart_equal(self.interval_squared(), 0, tol=tol)

    def __repr__(self):
        return f"FourWaveVector(omega/c={self.ct}, k=({self.vec[1]}, {self.vec[2]}, {self.vec[3]}))"
