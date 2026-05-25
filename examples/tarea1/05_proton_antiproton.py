"""Ejercicio 5: velocidad relativa protón-antiprotón."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import savefig, print_header


def relative_speed(beta: float) -> float:
    return (2 * beta) / (1 + beta**2)


def solve():
    print_header("Ejercicio 5: velocidad relativa")
    beta = 0.8
    rel = relative_speed(beta)
    print(f"Cada partícula: β={beta}")
    print(f"Velocidad relativa = {rel:.9f} c")
    return rel


def simulate_plot():
    solve()
    betas = np.linspace(0, 0.999, 600)
    rels = relative_speed(betas)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, rels)
    ax.scatter([0.8], [relative_speed(0.8)], zorder=3)
    ax.set_xlabel("β de cada partícula")
    ax.set_ylabel("β relativa")
    ax.set_title("Ejercicio 5: velocidades opuestas iguales")
    ax.grid(True)
    path = savefig(fig, "ejercicio_05_velocidad_relativa.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
