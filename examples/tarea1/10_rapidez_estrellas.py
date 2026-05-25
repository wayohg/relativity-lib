"""Ejercicio 10: rapidez y suma de velocidades."""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import velocity_addition_1d, savefig, print_header


def beta_after_n(beta_step: float, n: int) -> float:
    """Velocity after adding the same rapidity n times."""
    return math.tanh(n * math.atanh(beta_step))


def exact_beta_after_n(beta_step: float, n: int) -> float:
    """Exact expression using tanh(n atanh beta)."""
    ratio = (1.0 - beta_step) / (1.0 + beta_step)
    return (1.0 - ratio**n) / (1.0 + ratio**n)


def solve():
    print_header("Ejercicio 10: rapidez")

    print("(a) Si v=c tanh(u) y w=c tanh(U):")
    print("    (tanh u + tanh U)/(1 + tanh u tanh U) = tanh(u+U)")
    print("    por lo tanto w' = c tanh(u+U).")

    beta_step = 0.9
    rapidity_step = math.atanh(beta_step)
    print("(b) Para β=0.9 entre estrellas consecutivas:")
    print(f"    rapidez por paso u = artanh(0.9) = {rapidity_step:.9f}")
    print("    β_N = tanh(N artanh(0.9))")
    print("    forma exacta: β_N = (1 - (1/19)^N)/(1 + (1/19)^N)")
    print("    para N grande: β_N ≈ 1 - 2/19^N")

    for n in [1, 2, 3, 5, 10]:
        print(f"    N={n:2d}: β_N={beta_after_n(beta_step, n):.12f}")

    # Numerical check using repeated Einstein addition.
    beta_iter = 0.0
    for _ in range(10):
        beta_iter = velocity_addition_1d(beta_iter, beta_step)
    print(f"    Verificación iterativa N=10: β={beta_iter:.12f}")

    return beta_step


def simulate_plot():
    beta_step = solve()
    ns = np.arange(1, 16)
    betas = np.array([beta_after_n(beta_step, int(n)) for n in ns])
    approx = 1.0 - 2.0 / (19.0**ns)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, betas, marker="o", label=r"exacta $\tanh(N\operatorname{artanh}0.9)$")
    ax.plot(ns, approx, linestyle="--", label=r"aprox. $1-2/19^N$")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\beta_N$")
    ax.set_title("Ejercicio 10: velocidades por rapidez acumulada")
    ax.grid(True)
    ax.legend()
    path = savefig(fig, "ejercicio_10_rapidez_estrellas.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
