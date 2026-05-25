"""Ejercicio 1: marco en el que dos eventos son simultáneos."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import gamma_beta, savefig, print_header


def solve():
    print_header("Ejercicio 1: simultaneidad")

    # Use a=1 and c=1 for a dimensionless simulation.
    a = 1.0
    c = 1.0

    x1, t1 = a, 2 * a / c
    x2, t2 = 2 * a, 3 * a / (2 * c)

    dx = x2 - x1
    dt = t2 - t1

    # Simultaneity in K': Δt' = γ(Δt - v Δx/c²) = 0.
    v = c**2 * dt / dx
    beta = v / c
    g = gamma_beta(beta)

    dx_prime = g * (dx - v * dt)

    print(f"Δx = {dx:.3f} a")
    print(f"Δt = {dt:.3f} a/c")
    print(f"β = v/c = {beta:.6f}")
    print("K' debe moverse hacia -x respecto a K con rapidez 0.5c.")
    print(f"Separación espacial en K': Δx' = {dx_prime:.6f} a")

    return beta, dx_prime


def simulate_plot():
    a = 1.0
    x1, ct1 = a, 2 * a
    x2, ct2 = 2 * a, 1.5 * a
    beta, _ = solve()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([x1, x2], [ct1, ct2], zorder=3)
    ax.text(x1, ct1, "  Evento 1")
    ax.text(x2, ct2, "  Evento 2")

    # Lines of simultaneity in K' satisfy ct - beta*x = constant.
    xs = np.linspace(0.4, 2.4, 100)
    const = ct1 - beta * x1
    ax.plot(xs, beta * xs + const, label="línea t' constante")

    ax.set_xlabel("x/a")
    ax.set_ylabel("ct/a")
    ax.set_title("Ejercicio 1: eventos simultáneos en K'")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_01_simultaneidad.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
