from relativity import C, Event, ReferenceFrame, Particle, Photon, Collision
from relativity.sr.kinematics import lorentz_transform_event, time_dilation
from relativity.sr.doppler import longitudinal_doppler_frequency

# Usa c=1 para ejercicios teóricos rápidos.
c = 1.0
S = ReferenceFrame("S", [0, 0, 0], c=c)
Sp = ReferenceFrame("S'", [0.6*c, 0, 0], relative_to=S, c=c)

E = Event(t=2.0, r=[3.0, 0, 0], frame=S, c=c, name="A")
print("Evento en S':", E.in_frame(Sp))
print("Transformación directa:", lorentz_transform_event(2.0, [3, 0, 0], [0.6, 0, 0], c=c))

p = Particle(mass=2.0, velocity=[0.6*c, 0, 0], c=c, name="p")
print("gamma=", p.gamma, "E=", p.energy, "p=", p.momentum)

ph = Photon(frequency=5.0, direction=[1, 0, 0], c=c, h=1.0)
print("fotón lightlike:", ph.is_lightlike())
print("Doppler receding beta=0.5:", longitudinal_doppler_frequency(10.0, 0.5))

col = Collision([p], c=c)
print("masa invariante sistema:", col.invariant_mass())
