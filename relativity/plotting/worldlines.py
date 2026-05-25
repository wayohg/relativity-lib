"""Worldline plots in 1+1D."""
from __future__ import annotations


def plot_worldline(worldline, c=1.0, ax=None, x_index=0):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    x = [e.r[x_index] for e in worldline.events]
    ct = [c * e.t for e in worldline.events]
    ax.plot(x, ct, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("ct")
    return ax
