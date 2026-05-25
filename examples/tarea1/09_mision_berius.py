"""Ejercicio 9: tiempo de llegada en el marco de otra nave."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import gamma_beta, lorentz_x_event, velocity_transform_1d, savefig, print_header


def solve():
    print_header("Ejercicio 9: misión a Berius")

    L = 30.0       # ly in Earth frame
    beta_ara = 0.8
    beta_you = 0.6

    t_earth = L / beta_ara
    t_you, x_you = lorentz_x_event(t_earth, L, beta_you, c=1.0)
    beta_relative = velocity_transform_1d(beta_ara, beta_you)

    # Equivalent viewpoint in your frame at t'=0:
    # Earth-Berius distance is length-contracted: L/gamma_you.
    g_you = gamma_beta(beta_you)
    L_contracted = L / g_you
    closing_speed = beta_relative + beta_you
    t_you_alt = L_contracted / closing_speed

    print(f"Tiempo de llegada en Tierra: {t_earth:.6f} años")
    print(f"En tu marco por Lorentz: t' = {t_you:.6f} años")
    print(f"Posición del evento de llegada en tu marco: x' = {x_you:.6f} ly")
    print(f"Velocidad de ARA respecto a ti: {beta_relative:.9f} c")
    print(f"Verificación por cierre: {t_you_alt:.6f} años")

    return t_you


def simulate_plot():
    t_you = solve()
    L = 30.0
    beta_ara = 0.8
    beta_you = 0.6
    t_earth = L / beta_ara

    fig, ax = plt.subplots(figsize=(7, 5))
    ts = np.linspace(0, t_earth, 200)
    ax.plot(beta_ara * ts, ts, label="ARA en marco Tierra")
    ax.plot([L, L], [0, t_earth], label="Berius en marco Tierra")
    ax.plot(beta_you * ts, ts, label="tu nave en marco Tierra")
    ax.scatter([L], [t_earth], zorder=3, label="llegada")
    ax.set_xlabel("x [ly]")
    ax.set_ylabel("t [yr]")
    ax.set_title(f"Ejercicio 9: llegada; en tu marco t'={t_you:.3f} yr")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_09_berius.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
