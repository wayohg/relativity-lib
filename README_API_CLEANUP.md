# API cleanup layer

This patch replaces the flat `import *` public API in `sr/__init__.py` and
`plotting/__init__.py` with namespace-style imports.

## Why?

Several functions share names across modules, for example:

- `gamma`
- `beta`
- `momentum`
- `kinetic_energy`
- `four_momentum`

Using `from .kinematics import *` followed by `from .dynamics import *` can
silently overwrite names. The new API avoids that.

## Recommended usage

```python
from relativity.sr import kinematics, dynamics, doppler, decay

kinematics.gamma(beta_value=0.8)
dynamics.total_energy(1.0, [0.8, 0, 0], c=1)
```

For plotting:

```python
from relativity.plotting import kinematics, spacetime

kinematics.plot_gamma_vs_beta(show=True)
```

Direct imports from concrete modules still work:

```python
from relativity.sr.kinematics import gamma
from relativity.plotting.kinematics import plot_gamma_vs_beta
```

## Files included

```text
relativity/__init__.py
relativity/sr/__init__.py
relativity/plotting/__init__.py
relativity/math/__init__.py
relativity/physics/__init__.py
.gitignore
tests/test_public_api.py
```

## After copying

Run:

```bash
python -m pytest -q
python examples/run_all_examples.py
```
