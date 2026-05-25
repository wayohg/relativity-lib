# Capa de tests para `relativity`

Coloca la carpeta `tests/` en la raíz del proyecto, al mismo nivel que la carpeta `relativity/`.

Estructura esperada:

```text
mi_proyecto/
├── relativity/
│   ├── __init__.py
│   ├── constants.py
│   ├── utils.py
│   ├── math/
│   ├── physics/
│   ├── sr/
│   └── plotting/
└── tests/
```

Instala dependencias:

```bash
pip install pytest numpy sympy matplotlib
```

Ejecuta todo:

```bash
pytest -q
```

Ejecuta solo una parte:

```bash
pytest -q tests/test_sr_kinematics.py
pytest -q tests/test_physics_objects.py
pytest -q tests/test_plotting.py
```

## Sobre `test_known_regressions.py`

Ese archivo contiene pruebas de bugs ya detectados durante la revisión. Están marcadas con `pytest.mark.xfail`, así que documentan errores pendientes sin romper toda la suite.

Cuando corrijas uno de esos bugs, quita el `xfail` correspondiente para convertirlo en prueba normal.

## Qué cubren estos tests

- Importaciones generales.
- Utilidades híbridas numéricas/simbólicas.
- Métrica de Minkowski y transformaciones de Lorentz.
- Cinemática relativista.
- Dinámica, Doppler y decaimiento.
- Objetos físicos: `Event`, `FourVector`, `Particle`, `Photon`, `Worldline`, `ReferenceFrame`.
- Smoke tests de `plotting` usando backend `Agg` para no abrir ventanas.
- Regresiones conocidas marcadas como pendientes.
