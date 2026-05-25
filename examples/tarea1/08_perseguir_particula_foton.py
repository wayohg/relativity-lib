"""Ejercicio 8: perseguir una partícula y discusión sobre fotones."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import velocity_transform_1d, savefig, print_header


def solve():
    print_header("Ejercicio 8: perseguir una partícula")

    u = 0.95  # particle speed in S, c=1
    v = 0.60  # your speed in S, c=1
    u_prime = velocity_transform_1d(u, v)

    print(f"(a) Partícula: {u:.3f}c, tú: {v:.3f}c")
    print(f"    velocidad de la partícula respecto a ti = {u_prime:.9f} c")
    print("(b) No existe un marco inercial que se mueva a c.")
    print("    Por tanto, no puedes transformar al marco de un fotón.")
    print("    Todo observador inercial mide la luz con velocidad c.")

    return u_prime


def simulate_plot():
    solve()
    u = 0.95
    betas = np.linspace(0, 0.94, 500)
    u_primes = velocity_transform_1d(u, betas)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, u_primes)
    ax.scatter([0.6], [velocity_transform_1d(u, 0.6)], zorder=3)
    ax.set_xlabel("tu β")
    ax.set_ylabel("β de la partícula en tu marco")
    ax.set_title("Ejercicio 8: velocidad relativa al perseguir")
    ax.grid(True)
    path = savefig(fig, "ejercicio_08_perseguir.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
