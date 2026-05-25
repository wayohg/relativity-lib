"""Basic Minkowski diagram plotting."""
from __future__ import annotations


def plot_events(events, c=1.0, ax=None, labels=True):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    for i, e in enumerate(events):
        ax.scatter(e.x, c * e.t)
        if labels:
            ax.annotate(e.name or f"E{i}", (e.x, c * e.t))
    ax.set_xlabel("x")
    ax.set_ylabel("ct")
    ax.axline((0, 0), slope=1, linestyle="--")
    ax.axline((0, 0), slope=-1, linestyle="--")
    return ax
