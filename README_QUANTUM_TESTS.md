# Quantum tests

This layer adds tests for the new `relativity.quantum` package.

## Files

```text
tests/test_quantum_modules.py
```

## How to run

From the project root, with your `.venv` active:

```bash
python -m pytest tests/test_quantum_modules.py -q
```

Or run the full suite:

```bash
python -m pytest -q
```

These tests cover:

- photon energy, wavelength, momentum and zero-direction validation
- blackbody radiation and Wien's law
- photoelectric formulas and stopping-potential data
- Compton shift and backscatter limits
- De Broglie wavelength
- uncertainty relations
- continuous/discrete probability helpers for Tarea 3 exercise 10
- simple symbolic smoke tests
