"""
Spacetime and worldline plotting. Images are saved to examples/output.

Run from project root:
    python examples/08_plotting_worldline.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from relativity.physics.event import Event
from relativity.physics.worldline import Worldline
from relativity.plotting.spacetime import plot_spacetime_diagram
from relativity.plotting.worldline import plot_worldline_summary

from _helpers import output_dir, print_header


def main() -> None:
    print_header("08 Worldline plotting")

    c = 1.0
    events = [
        Event(0.0, [0.0, 0.0, 0.0], c=c),
        Event(1.0, [0.3, 0.0, 0.0], c=c),
        Event(2.0, [0.9, 0.0, 0.0], c=c),
        Event(3.0, [1.4, 0.0, 0.0], c=c),
    ]
    wl = Worldline(events, c=c)

    out = output_dir()

    fig, ax = plot_spacetime_diagram(
        events=events,
        event_labels=["A", "B", "C", "D"],
        worldlines=[wl],
        worldline_labels=["particle"],
        beta_axes=[0.5],
        c=c,
        show=False,
    )
    path = out / "spacetime_diagram.png"
    fig.savefig(path, dpi=150)
    print("saved:", path)

    fig, axes = plot_worldline_summary(wl, c=c, show=False)
    path = out / "worldline_summary.png"
    fig.savefig(path, dpi=150)
    print("saved:", path)


if __name__ == "__main__":
    main()
