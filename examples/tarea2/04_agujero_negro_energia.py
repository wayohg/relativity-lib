"""Ejercicio 4: energía total emitida por evaporación de un agujero negro."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea2_common import C_SI, print_header, savefig, fmt_sci


def main() -> None:
    print_header('Ejercicio 4 - Energía total de evaporación')

    mass_g = 1.0e15
    mass_kg = mass_g * 1.0e-3
    energy_j = mass_kg * C_SI**2

    print('Suponiendo que la evaporación convierte toda la masa en radiación:')
    print('E = M c^2')
    print(f'M = {fmt_sci(mass_g)} g = {fmt_sci(mass_kg)} kg')
    print(f'E = {fmt_sci(energy_j)} J')

    # Optional physical scale: equivalent TNT megatons.
    megaton_tnt_j = 4.184e15
    print(f'Equivalente aproximado: {energy_j / megaton_tnt_j:.6e} megatones de TNT')

    masses_g = np.logspace(10, 16, 400)
    energies_j = masses_g * 1e-3 * C_SI**2

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.loglog(masses_g, energies_j, label=r'$E=Mc^2$')
    ax.scatter([mass_g], [energy_j], zorder=3, label=r'$10^{15}\,\mathrm{g}$')
    ax.set_xlabel('Masa [g]')
    ax.set_ylabel('Energía [J]')
    ax.set_title('Energía total disponible por masa de agujero negro')
    ax.grid(True, which='both', alpha=0.35)
    ax.legend()
    path = savefig(fig, '04_agujero_negro_energia.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')


if __name__ == '__main__':
    main()
