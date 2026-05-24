import numpy as np
from relativity.physics.fourvector import FourVector
from relativity.math.minkowski import ETA

class Event:
    def __init__(self, t, r, frame, c):
        self.frame = frame
        self.fourvec = FourVector.from_time_space(t, r, c)
        self.c = c
    
    @property
    def t(self):
        return self.fourvec.vec[0] / self.c
    
    @property
    def position(self):
        return self.fourvec.vec[1:]
    
    @property
    def x(self):
        return self.fourvec.vec[1]
    
    @property
    def y(self):
        return self.fourvec.vec[2]
    
    @property
    def z(self):
        return self.fourvec.vec[3]
    
    def in_frame(self, new_frame):
        new_fourvec = self.frame._transform(self.fourvec, new_frame)
        return Event(new_fourvec.ct / self.c, new_fourvec.spatial, new_frame, self.c)

    #def in_frame(self, new_frame):
    #    Lambda = self.frame._transformation_to(new_frame)
    #    new_vec = Lambda @ self.fourvec.vec
    #    return Event(new_vec[0]/self.c, new_vec[1:], new_frame, self.c)

    def interval_squared(self):
        return self.fourvec.interval_squared()
    
    def proper_length_to(self, other):
        """
        Longitud propia entre dos eventos.
        Solo válida si el intervalo es space-like.
        """
        delta = other.fourvec.vec - self.fourvec.vec
        s2 = delta @ ETA @ delta
    
        if s2 >= 0:
            raise ValueError("Intervalo no es space-like.")
    
        return smart_sqrt(-s2) / self.c
    
    def proper_time_to(self, other):
        """
        Tiempo propio entre dos eventos (invariante).
        """
        delta = other.fourvec.vec - self.fourvec.vec
        s2 = delta @ ETA @ delta

        if s2 < 0:
            raise ValueError("Intervalo tipo espacio: no hay tiempo propio.")

        return smart_sqrt(s2) / self.c
    
    def simultaneous_in_frame(self, other, frame):
        delta = other.fourvec.vec - self.fourvec.vec
        u = frame.four_velocity()
        return np.isclose(np.dot(u, delta), 0)