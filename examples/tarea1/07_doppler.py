"""Ejercicio 7: efecto Doppler relativista."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import longitudinal_doppler_factor, savefig, print_header


def solve():
    print_header("Ejercicio 7: Doppler relativista")

    f = 300.0
    beta = 0.4
    factor_receding = longitudinal_doppler_factor(beta, approaching=False)
    factor_approaching = longitudinal_doppler_factor(beta, approaching=True)

    f_receding = f * factor_receding
    f_approaching = f * factor_approaching

    print(f"f emitida/medida en S = {f:.3f} Hz")
    print(f"Observador S' moviéndose en la misma dirección que la luz, β={beta}:")
    print(f"  f' = {f_receding:.6f} Hz -> redshift")
    print("Si el observador viene hacia la luz:")
    print(f"  f' = {f_approaching:.6f} Hz -> blueshift")

    return f_receding, f_approaching


def simulate_plot():
    solve()
    f = 300.0
    betas = np.linspace(0, 0.95, 500)
    red = np.array([f * longitudinal_doppler_factor(b, approaching=False) for b in betas])
    blue = np.array([f * longitudinal_doppler_factor(b, approaching=True) for b in betas])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, red, label="misma dirección: redshift")
    ax.plot(betas, blue, label="hacia la luz: blueshift")
    ax.scatter([0.4, 0.4], [f * longitudinal_doppler_factor(0.4, False), f * longitudinal_doppler_factor(0.4, True)], zorder=3)
    ax.set_xlabel("β")
    ax.set_ylabel("frecuencia observada [Hz]")
    ax.set_title("Ejercicio 7: Doppler longitudinal")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_07_doppler.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
