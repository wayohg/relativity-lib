import numpy as np

from relativity.physics.fourvector import FourVector
from ..math.lorentz import boost_matrix
from relativity.math import relativistic_velocity_addition

class ReferenceFrame:
    def __init__(self, name, velocity, relative_to = None, c=1.0):
        self.name = name
        self.v = smart_array(velocity, dtype=float)
        self.c = float(c)
        self.relative_to = relative_to

    @property
    def velocity(self):
        return self.v
    
    @property
    def beta(self):
        return self.v / self.c
    
    def _velocity_wrt_root(self):
        if self.relative_to is None:
            return self.v

        parent_v = self.relative_to._velocity_wrt_root()

        return relativistic_velocity_addition(self.v, parent_v, self.c)
    
    def velocity_wrt(self, other_frame):
        """
        Velocidad de otro marco vista desde este.
        """
        v_self_root = self._velocity_wrt_root()
        v_other_root = other_frame._velocity_wrt_root()

        return relativistic_velocity_addition(v_self_root, -v_other_root, self.c)
        

    def _transformation_to(self, other):
        """
        Matriz que transforma coordenadas de ESTE marco al otro.
        """
        v_rel = other.velocity_wrt(self)
        return boost_matrix(v_rel, self.c)
    
    def _transform(self, fourvector, other):
        Lambda = self._transformation_to(other)
        new_vec = Lambda @ fourvector.vec
        return FourVector(*new_vec)

    #def _transform(self, fourvector, other):
    #    Lambda = self._transformation_to(other)
    #    return Lambda @ fourvector.vec

    def four_velocity(self):
        v2 = np.dot(self.v, self.v)
        if v2 == 0:
            return FourVector(self.c, 0.0, 0.0, 0.0)
        gamma = 1.0 / smart_sqrt(1 - v2/self.c**2)
        return FourVector(gamma*self.c, *(gamma*self.v))