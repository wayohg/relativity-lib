# relativity

Paquete inicial para ejercicios y simulaciones de relatividad especial.

## Diseño

- `physics/`: objetos físicos (`Event`, `ReferenceFrame`, `Particle`, `Photon`, `Worldline`).
- `math/`: geometría de Minkowski, boosts de Lorentz y tensores.
- `sr/`: fórmulas de relatividad especial listas para resolver ejercicios (`kinematics`, `dynamics`, `collision`, `doppler`, `decay`).
- `plotting/`: diagramas de Minkowski y worldlines.

## Uso rápido

```python
from relativity import Event, ReferenceFrame

c = 1.0
S = ReferenceFrame("S", [0,0,0], c=c)
Sp = ReferenceFrame("S'", [0.6,0,0], relative_to=S, c=c)
E = Event(t=2, r=[3,0,0], frame=S, c=c)
print(E.in_frame(Sp))
```

Recomendación: para ejercicios teóricos usa `c=1`. Para problemas de laboratorio usa `relativity.constants.C`.
