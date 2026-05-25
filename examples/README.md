# Examples for `relativity`

Coloca esta carpeta `examples/` al mismo nivel que `relativity/`, `tests/`, `pytest.ini` y `pyproject.toml`.

Estructura sugerida:

```text
Fisica moderna/
├── relativity/
├── tests/
├── examples/
├── pyproject.toml
└── requirements-dev.txt
```

Ejecuta ejemplos individuales desde la raíz del proyecto:

```bash
python examples/01_basic_kinematics.py
python examples/02_symbolic_kinematics.py
python examples/03_particles_and_photons.py
python examples/04_events_worldlines.py
python examples/05_dynamics_energy_momentum.py
python examples/06_doppler_decay.py
python examples/07_plotting_basic.py
python examples/08_plotting_worldline.py
```

O ejecuta todos:

```bash
python examples/run_all_examples.py
```

Los ejemplos de gráficas guardan imágenes en:

```text
examples/output/
```

Recomendación: antes de probar ejemplos, asegúrate de que los tests pasen:

```bash
python -m pytest -q
```
