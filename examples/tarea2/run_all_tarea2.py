"""Run all Tarea 2 scripts in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    '01_fuerza_perpendicular.py',
    '02_tevatr_protones.py',
    '03_derivacion_beta.py',
    '04_agujero_negro_energia.py',
    '05_aniquilacion_electron_positron.py',
    '06_decaimiento_kaon.py',
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n>>> Running {script}")
        subprocess.run([sys.executable, str(here / script)], check=True)
    print('\nAll Tarea 2 scripts finished successfully.')


if __name__ == '__main__':
    main()
