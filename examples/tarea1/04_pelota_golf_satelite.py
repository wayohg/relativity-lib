"""Ejercicio 4: contracción de longitud y tiempos de viaje."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import C_SI, gamma_beta, savefig, print_header


def solve():
    print_header("Ejercicio 4: pelota de golf a 0.94c")

    beta = 0.94
    g = gamma_beta(beta)
    L_km = 3.58e4
    L_m = L_km * 1000.0

    L_ball_km = L_km / g
    t_earth_s = L_m / (beta * C_SI)
    t_ball_s = t_earth_s / g

    print(f"β = {beta:.3f}, γ = {g:.9f}")
    print(f"(a) Distancia medida en la pelota = {L_ball_km:.6f} km")
    print(f"(b) Tiempo Tierra = {t_earth_s:.6f} s")
    print(f"    Tiempo pelota = {t_ball_s:.6f} s")

    return L_ball_km, t_earth_s, t_ball_s


def simulate_plot():
    solve()
    betas = np.linspace(0.01, 0.99, 600)
    L_km = 3.58e4
    contracted = L_km / gamma_beta(betas)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, contracted)
    ax.axvline(0.94, linestyle="--", label="β=0.94")
    ax.set_xlabel("β = v/c")
    ax.set_ylabel("distancia medida en la pelota [km]")
    ax.set_title("Ejercicio 4: contracción de la distancia al satélite")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_04_contraccion_satelite.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
