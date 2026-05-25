"""
Run all examples in order.

Run from project root:
    python examples/run_all_examples.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EXAMPLES = [
    "01_basic_kinematics.py",
    "02_symbolic_kinematics.py",
    "03_particles_and_photons.py",
    "04_events_worldlines.py",
    "05_dynamics_energy_momentum.py",
    "06_doppler_decay.py",
    "07_plotting_basic.py",
    "08_plotting_worldline.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    failed = []

    for script in EXAMPLES:
        path = here / script
        print(f"\n>>> Running {script}")
        result = subprocess.run([sys.executable, str(path)], check=False)
        if result.returncode != 0:
            failed.append(script)

    if failed:
        print("\nFailed examples:")
        for item in failed:
            print("-", item)
        raise SystemExit(1)

    print("\nAll examples ran successfully.")


if __name__ == "__main__":
    main()
