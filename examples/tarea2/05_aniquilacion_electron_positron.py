"""Ejercicio 5: aniquilación electrón-positrón en dos fotones."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _tarea2_common import (
    ELECTRON_REST_ENERGY_MEV,
    gamma_beta,
    photon_frequency_from_energy_MeV,
    print_header,
    savefig,
    fmt_sci,
)


def main() -> None:
    print_header('Ejercicio 5 - Aniquilación electrón-positrón')

    beta = 0.6
    gamma = gamma_beta(beta)
    E_each_particle_MeV = gamma * ELECTRON_REST_ENERGY_MEV
    E_total_MeV = 2.0 * E_each_particle_MeV
    E_each_photon_MeV = E_total_MeV / 2.0
    frequency = photon_frequency_from_energy_MeV(E_each_photon_MeV)

    print(f'beta = {beta}')
    print(f'gamma = {gamma:.8f}')
    print(f'Energía de cada partícula inicial = gamma m_e c^2 = {E_each_particle_MeV:.8f} MeV')
    print(f'Energía total inicial = {E_total_MeV:.8f} MeV')
    print(f'Como el momento total inicial es cero, los dos fotones salen opuestos y con igual energía.')
    print(f'Energía de cada fotón = {E_each_photon_MeV:.8f} MeV')
    print(f'Frecuencia de cada fotón = {fmt_sci(frequency)} Hz')
    print('\n¿Por qué no un solo fotón?')
    print('El momento total inicial es cero. Un único fotón con energía no nula tiene |p| = E/c,')
    print('por lo tanto no puede tener momento total cero. Se violaría la conservación del momento.')

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    # Initial momenta
    ax.arrow(-0.15, 0.6, -0.75, 0, head_width=0.06, length_includes_head=True)
    ax.arrow(0.15, 0.6, 0.75, 0, head_width=0.06, length_includes_head=True)
    ax.text(-0.9, 0.72, r'$e^-$')
    ax.text(0.75, 0.72, r'$e^+$')
    # Final photons
    ax.arrow(0.0, -0.2, -0.9, 0, head_width=0.06, length_includes_head=True)
    ax.arrow(0.0, -0.2, 0.9, 0, head_width=0.06, length_includes_head=True)
    ax.text(-0.95, -0.08, r'$\gamma$')
    ax.text(0.85, -0.08, r'$\gamma$')
    ax.text(0, 0.25, 'momento total = 0', ha='center')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.0)
    ax.axis('off')
    ax.set_title('Aniquilación con conservación de momento')
    path = savefig(fig, '05_aniquilacion_momento.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')


if __name__ == '__main__':
    main()
