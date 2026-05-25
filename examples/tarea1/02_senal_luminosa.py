"""Ejercicio 2: señal luminosa y transformación de Lorentz."""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from _tarea1_common import C_SI, gamma_beta, savefig, print_header


def solve():
    print_header("Ejercicio 2: señal luminosa")

    x, y, z = 3.0, 5.0, 10.0
    r = math.sqrt(x**2 + y**2 + z**2)
    t = r / C_SI
    beta = 0.8
    g = gamma_beta(beta)

    # For boost along x. Use ct in meters.
    ct = r
    ct_p = g * (ct - beta * x)
    x_p = g * (x - beta * ct)
    y_p = y
    z_p = z
    t_p = ct_p / C_SI

    r_p = math.sqrt(x_p**2 + y_p**2 + z_p**2)
    speed_p = r_p / t_p

    print(f"(a) Distancia recorrida = sqrt(134) = {r:.9f} m")
    print(f"    t = r/c = {t:.12e} s")
    print(f"(b) Para β=0.8, γ={g:.6f}")
    print(f"    x' = {x_p:.9f} m")
    print(f"    y' = {y_p:.9f} m")
    print(f"    z' = {z_p:.9f} m")
    print(f"    t' = {t_p:.12e} s")
    print(f"(c) |r'|/t' = {speed_p:.6f} m/s")
    print(f"    c        = {C_SI:.6f} m/s")
    print(f"    error relativo = {abs(speed_p - C_SI)/C_SI:.3e}")

    return x_p, y_p, z_p, t_p, speed_p


def simulate_plot():
    x, y, z, t, _ = solve()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([0, 3], [0, 5], [0, 10], marker="o", label="Trayectoria en K")
    ax.plot([0, x], [0, y], [0, z], marker="o", label="Evento recepción en K'")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Ejercicio 2: señal luminosa")
    ax.legend()
    path = savefig(fig, "ejercicio_02_senal_luminosa.png")
    print(f"Figura guardada en: {path}")


if __name__ == "__main__":
    simulate_plot()
