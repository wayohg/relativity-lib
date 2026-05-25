"""Run all Tarea 1 solution/simulation scripts."""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "01_eventos_simultaneos.py",
    "02_senal_luminosa.py",
    "03_viaje_astronautas.py",
    "04_pelota_golf_satelite.py",
    "05_proton_antiproton.py",
    "06_intervalo_spacelike.py",
    "07_doppler.py",
    "08_perseguir_particula_foton.py",
    "09_mision_berius.py",
    "10_rapidez_estrellas.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n>>> Ejecutando {script}")
        runpy.run_path(str(HERE / script), run_name="__main__")

    print("\nTodos los scripts de Tarea 1 terminaron correctamente.")
    print(f"Figuras generadas en: {HERE / 'output'}")


if __name__ == "__main__":
    main()
