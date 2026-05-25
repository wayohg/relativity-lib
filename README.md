# Relativity hybrid improved

Versión con cambios para trabajar con números `float`/NumPy y expresiones simbólicas de SymPy sin mantener un `symbolic.py` separado.

## Cambios principales

- `utils.py` ahora es el backend híbrido central.
- Se eliminaron conversiones internas innecesarias a `dtype=float` en módulos físicos/matemáticos.
- Se reemplazaron operaciones directas como `np.dot`, `np.sqrt` y `np.linalg.norm` por helpers `smart_*` donde era importante.
- `math/lorentz.py` acepta boosts simbólicos y numéricos.
- `physics/event.py`, `frame.py`, `fourvector.py`, `radiation.py` y `worldline.py` ya no fuerzan `float`.
- `sr/kinematics.py` incluido.
- `Particle` y `Photon` conservan la API anterior y agregan constructores alternativos.

## Uso sugerido

Copia la carpeta `relativity/` sobre tu proyecto, o compara archivo por archivo antes de reemplazar.

Ejemplo numérico:

```python
from relativity.constants import C
from relativity.physics import Particle

p = Particle(mass=1.0, velocity=[0.8*C, 0, 0], c=C)
print(p.gamma)
print(p.energy)
print(p.momentum)
```

Ejemplo simbólico:

```python
import sympy as sp
from relativity.physics import Particle

m, v, c = sp.symbols("m v c", positive=True)
p = Particle(mass=m, velocity=[v, 0, 0], c=c)

print(p.gamma)
print(p.energy)
print(p.momentum)
```

## Recomendación

No reemplaces todo a ciegas si ya tienes avances locales. Primero descomprime este ZIP en una carpeta aparte y compara con tu versión usando VS Code, Meld, `diff`, Git, etc.
