"""Ejercicio 6: K0 -> pi0 + pi0."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea2_common import (
    gamma_beta,
    momentum_pc_from_energy_mass,
    velocity_addition_beta,
    print_header,
    savefig,
)


def main() -> None:
    print_header('Ejercicio 6 - Decaimiento K0 -> pi0 + pi0')

    beta_K_lab = 0.9
    M_K_MeV = 498.0
    m_pi_MeV = 135.0

    Q_MeV = M_K_MeV - 2.0 * m_pi_MeV
    E_pi_star_MeV = M_K_MeV / 2.0
    p_pi_star_c_MeV = momentum_pc_from_energy_mass(E_pi_star_MeV, m_pi_MeV)
    beta_pi_star = p_pi_star_c_MeV / E_pi_star_MeV

    beta_forward_lab = velocity_addition_beta(beta_pi_star, beta_K_lab)
    beta_backward_lab = velocity_addition_beta(-beta_pi_star, beta_K_lab)

    print('Marco de reposo inicial del kaón:')
    print(f'M_K c^2 = {M_K_MeV:.6f} MeV')
    print(f'm_pi c^2 = {m_pi_MeV:.6f} MeV')
    print(f'(a) Energía liberada Q = M_K c^2 - 2 m_pi c^2 = {Q_MeV:.6f} MeV')
    print(f'Energía de cada pion: E_pi* = {E_pi_star_MeV:.6f} MeV')
    print(f'(b) Magnitud del momento de cada pion: p* c = {p_pi_star_c_MeV:.6f} MeV')
    print('Los momentos son opuestos: +p* y -p*.')
    print(f'beta_pi* = v_pi*/c = {beta_pi_star:.8f}')
    print('\nPara (c), se asume que un pion sale en la dirección del movimiento del kaón')
    print('y el otro en la dirección opuesta. Si el ángulo cambia, las velocidades de laboratorio cambian.')
    print(f'beta_K_lab = {beta_K_lab:.6f}, gamma_K = {gamma_beta(beta_K_lab):.6f}')
    print(f'Pion hacia adelante: beta_lab = {beta_forward_lab:.8f}')
    print(f'Pion hacia atrás:    beta_lab = {beta_backward_lab:.8f}')

    # Plot the angle dependence of laboratory speed to make the caveat explicit.
    theta = np.linspace(0, np.pi, 500)
    bx_star = beta_pi_star * np.cos(theta)
    by_star = beta_pi_star * np.sin(theta)
    denom = 1 + bx_star * beta_K_lab
    bx_lab = (bx_star + beta_K_lab) / denom
    by_lab = by_star / (gamma_beta(beta_K_lab) * denom)
    speed_lab = np.sqrt(bx_lab**2 + by_lab**2)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(theta, speed_lab, label=r'$|\beta_{\pi,lab}|$')
    ax.scatter([0, np.pi], [speed_lab[0], speed_lab[-1]], zorder=3, label='casos colineales')
    ax.set_xlabel(r'Ángulo $\theta^*$ en el marco del kaón [rad]')
    ax.set_ylabel(r'$|v_{lab}|/c$')
    ax.set_title('Velocidad de los piones en el laboratorio')
    ax.grid(True, alpha=0.35)
    ax.legend()
    path = savefig(fig, '06_decaimiento_kaon_velocidades_lab.png')
    plt.close(fig)
    print(f'Figura guardada: {path}')


if __name__ == '__main__':
    main()
