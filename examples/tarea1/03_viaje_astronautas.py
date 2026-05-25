"""Ejercicio 3: viaje de ida y vuelta a una estrella a 20 años luz."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import gamma_beta, savefig, print_header


def solve():
    print_header("Ejercicio 3: viaje relativista")

    distance_one_way_ly = 20.0
    max_proper_years = 40.0

    # Earth time round trip: T = 2L/(βc) = 40/β yr in ly/yr units.
    # Proper time: τ = T/γ <= 40 -> 1/(βγ) <= 1 -> βγ >= 1.
    beta_min = 1.0 / np.sqrt(2.0)
    gamma_min = gamma_beta(beta_min)
    earth_time = 2 * distance_one_way_ly / beta_min
    proper_time = earth_time / gamma_min

    print(f"β mínimo = {beta_min:.9f}")
    print(f"v mínimo = {beta_min:.9f} c")
    print(f"γ = {gamma_min:.9f}")
    print(f"Tiempo de la Tierra = {earth_time:.9f} años")
    print(f"Tiempo propio astronautas = {proper_time:.9f} años")

    return beta_min, earth_time, proper_time


def simulate_plot():
    beta_min, _, _ = solve()
    betas = np.linspace(0.1, 0.999, 700)
    earth_times = 40.0 / betas
    proper_times = earth_times / gamma_beta(betas)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, proper_times, label="tiempo astronautas")
    ax.plot(betas, earth_times, label="tiempo Tierra", alpha=0.75)
    ax.axhline(40.0, linestyle="--", label="límite 40 años")
    ax.axvline(beta_min, linestyle=":", label=fr"β mínimo={beta_min:.3f}")
    ax.set_xlabel("β = v/c")
    ax.set_ylabel("tiempo [años]")
    ax.set_title("Ejercicio 3: tiempo de viaje vs velocidad")
    ax.set_ylim(0, 120)
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_03_viaje_astronautas.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
