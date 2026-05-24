import numpy as np
from .fourvector import FourVector
from ..math.minkowski import minkowski_dot

class FourWaveVector(FourVector):
    """
    4-vector onda para fotones.
    k^μ = (ω/c, kx, ky, kz)
    """
    def __init__(self, omega_c, k_vec):
        # Initialize the parent FourVector with unpacked 3D spatial components
        super().__init__(omega_c, *k_vec)

    def frequency_measured_by(self, frame):
        """
        Frecuencia angular medida por un observador
        en el marco dado.
        ω = - k · u
        """
        u = frame.four_velocity()
        return minkowski_dot(self.vec, u.vec)
    
    def redshift(k, emitter_frame, observer_frame):
        """
        Calcula el corrimiento al rojo.
        """
        omega_emit = k.frequency_measured_by(emitter_frame)
        omega_obs  = k.frequency_measured_by(observer_frame)
    
        return omega_emit / omega_obs - 1.0