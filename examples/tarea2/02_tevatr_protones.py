"""Ejercicio 2: rapidez de protones en las etapas del Tevatrón."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea2_common import (
    PROTON_REST_ENERGY_MEV,
    beta_from_kinetic_energy,
    gamma_beta,
    print_header,
    savefig,
)


def main() -> None:
    print_header('Ejercicio 2 - Rapidez de protones en el Tevatrón')

    stages = [
        ('Cockcroft-Walton', 0.750),       # 750 keV = 0.750 MeV
        ('Linac', 400.0),
        ('Booster', 8_000.0),
        ('Inyector principal', 150_000.0),
        ('Tevatrón', 1_000_000.0),
    ]

    print(f'Energía en reposo del protón: E0 = {PROTON_REST_ENERGY_MEV:.6f} MeV')
    print('\n{:<22s} {:>15s} {:>15s} {:>15s}'.format('Etapa', 'K [MeV]', 'beta=v/c', 'gamma'))
    print('-' * 72)

    betas = []
    gammas = []
    for name, K in stages:
        beta = beta_from_kinetic_energy(K, PROTON_REST_ENERGY_MEV)
        gamma = gamma_beta(beta)
        betas.append(beta)
        gammas.append(gamma)
        print(f'{name:<22s} {K:15.6g} {beta:15.10f} {gamma:15.6f}')

    x = np.arange(len(stages))
    labels = [s[0] for s in stages]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(x, betas, marker='o', label=r'$\beta$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel(r'$\beta=v/c$')
    ax.set_title('Rapidez relativista al final de cada etapa')
    ax.grid(True, alpha=0.35)
    ax.legend()
    path = savefig(fig, '02_tevatr_betas.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.semilogy(x, np.array(gammas), marker='o', label=r'$\gamma$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel(r'$\gamma$')
    ax.set_title('Factor de Lorentz por etapa')
    ax.grid(True, alpha=0.35)
    ax.legend()
    path = savefig(fig, '02_tevatr_gammas.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')


if __name__ == '__main__':
    main()
