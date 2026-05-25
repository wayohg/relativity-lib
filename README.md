# Relativity

A Python library for solving, simulating, and visualizing problems in **special relativity**.

The project is designed to work in a hybrid way: most core functions can operate with both ordinary numerical values (`float`, `int`, NumPy arrays) and symbolic expressions from SymPy.

## Main goals

- Solve standard special relativity exercises.
- Work numerically and symbolically with the same API when possible.
- Represent physical objects such as particles, photons, events, frames, and worldlines.
- Plot relativistic effects, spacetime diagrams, trajectories, dynamics, and decay processes.
- Provide a clean structure that can grow into simulations, animations, and educational notebooks.

---

## Project structure

```text
relativity/
├── constants.py
├── utils.py
│
├── math/
│   ├── minkowski.py
│   ├── lorentz.py
│   └── tensors.py
│
├── physics/
│   ├── event.py
│   ├── frame.py
│   ├── fourvector.py
│   ├── particle.py
│   ├── photon.py
│   ├── worldline.py
│   └── radiation.py
│
├── sr/
│   ├── kinematics.py
│   ├── dynamics.py
│   ├── doppler.py
│   ├── decay.py
│   └── collision.py
│
└── plotting/
    ├── style.py
    ├── utils.py
    ├── spacetime.py
    ├── worldline.py
    ├── kinematics.py
    ├── dynamics.py
    └── decay.py
```

Additional project folders:

```text
tests/      pytest test suite
examples/   runnable examples and plotting demos
```

---

## Installation for development

From the root of the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The editable installation is recommended during development because changes inside `relativity/` are immediately reflected when running tests or examples.

---

## Quick start

### Basic kinematics

```python
from relativity.constants import C
from relativity.sr.kinematics import gamma, dilated_time, contracted_length

v = [0.8 * C, 0, 0]

print(gamma(v))
print(dilated_time(1.0, v))
print(contracted_length(10.0, v))
```

### Symbolic calculations

```python
import sympy as sp
from relativity.sr.kinematics import gamma, lorentz_transform_event_1d

v, c = sp.symbols("v c", positive=True)

print(gamma([v, 0, 0], c=c))

x, t = sp.symbols("x t")
t_prime, x_prime = lorentz_transform_event_1d(t, x, v, c=c)

print(t_prime)
print(x_prime)
```

### Particles and photons

```python
from relativity.constants import C
from relativity.physics import Particle, Photon

particle = Particle(mass=1.0, velocity=[0.6 * C, 0, 0])
photon = Photon(frequency=5e14, direction=[1, 0, 0])

print(particle.gamma)
print(particle.energy)
print(photon.energy)
print(photon.momentum)
```

### Events and worldlines

```python
from relativity.physics import Event, Worldline

A = Event(t=0, r=[0, 0, 0], c=1)
B = Event(t=1, r=[0.5, 0, 0], c=1)
C = Event(t=2, r=[1.0, 0, 0], c=1)

wl = Worldline([A, B, C], c=1)

print(wl.velocities())
print(wl.proper_time())
```

---

## Plotting examples

```python
from relativity.plotting.kinematics import plot_gamma_vs_beta

plot_gamma_vs_beta(show=True)
```

```python
from relativity.physics import Event, Worldline
from relativity.plotting.spacetime import plot_spacetime_diagram

A = Event(t=0, r=[0, 0, 0], c=1)
B = Event(t=1, r=[0.5, 0, 0], c=1)
wl = Worldline([A, B], c=1)

plot_spacetime_diagram(
    events=[A, B],
    event_labels=["A", "B"],
    worldlines=[wl],
    worldline_labels=["particle"],
    c=1,
    show=True,
)
```

---

## Public API style

The public API avoids broad `import *` inside package `__init__.py` files to prevent silent name collisions.

Recommended imports:

```python
from relativity.sr import kinematics, dynamics, doppler, decay

kinematics.gamma(beta_value=0.8)
dynamics.total_energy(1.0, [0.8, 0, 0], c=1)
```

Direct module imports are also recommended:

```python
from relativity.sr.kinematics import gamma
from relativity.plotting.kinematics import plot_gamma_vs_beta
```

Avoid relying on ambiguous imports such as:

```python
from relativity.sr import gamma
```

because different modules may define similar concepts such as `gamma`, `momentum`, or `kinetic_energy`.

---

## Running tests

From the root of the project:

```bash
python -m pytest -q
```

A healthy development state should report all tests passing.

Current validated state:

```text
52 passed
```

---

## Running examples

From the root of the project:

```bash
python examples/run_all_examples.py
```

Individual examples can also be executed:

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

Plotting examples save images inside:

```text
examples/output/
```

This folder should not be tracked by Git.

---

## Development notes

### Hybrid numeric/symbolic design

The core idea is to avoid forcing `dtype=float` unless numerical evaluation is explicitly needed. Use helpers from `relativity.utils` instead of raw NumPy operations when symbolic compatibility matters.

Prefer:

```python
smart_array(v)
smart_dot(a, b)
smart_norm(v)
smart_sqrt(x)
normalize_vector(direction)
```

Avoid inside core symbolic-compatible code:

```python
np.array(v, dtype=float)
np.dot(a, b)
np.linalg.norm(v)
np.sqrt(x)
direction / np.linalg.norm(direction)
```

### Validation with symbolic expressions

Numerical comparisons should not be applied blindly to symbolic expressions.

Prefer:

```python
if not is_symbolic(beta2) and beta2 >= 1:
    raise ValueError("Speed must satisfy v < c.")
```

instead of:

```python
if beta2 >= 1:
    raise ValueError("Speed must satisfy v < c.")
```

---

## Suggested next steps

Recommended development order:

1. Add `plotting/animation.py`.
2. Add more applied examples:
   - Lorentz transformations
   - Twin paradox
   - Muon decay length
   - Relativistic collisions
3. Add `tests/test_examples_run.py` to automatically check that examples execute.
4. Expand documentation in `docs/`.
5. Review and improve `sr/collision.py`.

---

## Git ignore recommendation

A typical `.gitignore` should include:

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
examples/output/
build/
dist/
*.egg-info/
```

---

## Status

The project currently has:

- A hybrid symbolic/numeric core.
- Physics objects for events, frames, worldlines, particles, and photons.
- Special relativity modules for kinematics, dynamics, Doppler shift, decay, and collisions.
- Plotting tools for spacetime diagrams, worldlines, kinematics, dynamics, and decay.
- A working pytest suite.
- Runnable examples.

