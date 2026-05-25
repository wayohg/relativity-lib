"""Ejercicio 6: intervalo tipo espacio y simultaneidad."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import gamma_beta, savefig, print_header


def solve():
    print_header("Ejercicio 6: relación spacelike")

    # Use a=1 and c=1.
    dx = 1.0
    dt = 0.5
    s2 = dt**2 - dx**2
    beta = dt / dx
    g = gamma_beta(beta)
    dx_prime = g * (dx - beta * dt)

    print(f"Δx = a, cΔt = a/2")
    print(f"s² = c²Δt² - Δx² = {s2:.6f} a²")
    print("Como s² < 0, la separación es tipo espacio.")
    print(f"β para simultaneidad = {beta:.6f}")
    print(f"Δx' = {dx_prime:.6f} a")

    return s2, beta, dx_prime


def simulate_plot():
    _, beta, _ = solve()
    x1, ct1 = 1.0, 0.5
    x2, ct2 = 2.0, 1.0
    xs = np.linspace(0.5, 2.5, 100)
    const = ct1 - beta * x1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([x1, x2], [ct1, ct2], zorder=3)
    ax.plot(xs, beta * xs + const, label="t' constante")
    ax.plot(xs, xs, linestyle=":", label="cono de luz ct=x")
    ax.set_xlabel("x/a")
    ax.set_ylabel("ct/a")
    ax.set_title("Ejercicio 6: eventos tipo espacio")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_06_spacelike.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
