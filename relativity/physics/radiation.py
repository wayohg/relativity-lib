import numpy as np
from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import minkowski_dot
from relativity.utils import smart_array


class FourWaveVector(FourVector):
    """
    Four-wave-vector for photons/radiation.
    k^μ = (ω/c, k⃗) where |k⃗| = ω/c.
    """

    def __init__(self, omega_c, k_vec):
        k_vec = smart_array(k_vec, dtype=float)
        super().__init__(omega_c, *k_vec)

    # =====================================================
    # FREQUENCY MEASUREMENT
    # =====================================================

    def frequency_measured_by(self, frame):
        """
        Angular frequency measured by an observer in frame.
        ω = k · U  (with +--- signature, ω = k^μ U_μ = k^0 U^0 - k⃗·u⃗)
        FIX: original had wrong sign; Minkowski dot with +--- gives positive ω
        """
        u = frame.four_velocity()
        return minkowski_dot(self.vec, u.vec)

    # =====================================================
    # REDSHIFT
    # =====================================================

    @staticmethod                                     # FIX: missing self → made staticmethod
    def redshift(k, emitter_frame, observer_frame):
        """
        Gravitational/kinematic redshift z = ω_emit/ω_obs - 1.
        """
        omega_emit = k.frequency_measured_by(emitter_frame)
        omega_obs  = k.frequency_measured_by(observer_frame)

        if omega_obs == 0:
            raise ValueError("Observer frequency is zero; cannot compute redshift.")

        return omega_emit / omega_obs - 1.0

    # =====================================================
    # NULL CHECK
    # =====================================================

    def is_lightlike(self, tol=1e-10):
        """A valid wave vector must be null: k·k = 0."""
        return abs(self.interval_squared()) < tol

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        return (
            f"FourWaveVector(ω/c={self.ct:.4g}, "
            f"k=({self.vec[1]:.4g}, {self.vec[2]:.4g}, {self.vec[3]:.4g}))"
        )
