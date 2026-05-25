"""Ejercicio 1: fuerza perpendicular a la velocidad."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea2_common import print_header, gamma_beta, savefig


def main() -> None:
    print_header('Ejercicio 1 - Fuerza perpendicular')
    print('Partimos de p = gamma m v y F = dp/dt.')
    print('dp/dt = m (gamma a + v dgamma/dt).')
    print('Como dgamma/dt = gamma^3 (v·a)/c^2, si F siempre es perpendicular a v')
    print('entonces F·v = dE/dt = 0, la rapidez no cambia y v·a = 0.')
    print('Por tanto dgamma/dt = 0 y queda: F = m gamma a.')

    beta = np.linspace(0.0, 0.99, 400)
    gamma = gamma_beta(beta)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(beta, gamma, label=r'$F_\perp/(ma)=\gamma$')
    ax.set_xlabel(r'$\beta=v/c$')
    ax.set_ylabel(r'Factor transversal $\gamma$')
    ax.set_title('Masa inercial transversal efectiva')
    ax.grid(True, alpha=0.35)
    ax.legend()
    path = savefig(fig, '01_fuerza_perpendicular_gamma.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')


if __name__ == '__main__':
    main()
