"""
Basic plotting examples. Images are saved to examples/output.

Run from project root:
    python examples/07_plotting_basic.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from relativity.plotting.kinematics import plot_kinematics_summary
from relativity.plotting.dynamics import plot_dynamics_summary
from relativity.plotting.decay import plot_decay_summary

from _helpers import output_dir, print_header


def main() -> None:
    print_header("07 Basic plotting")

    out = output_dir()

    fig, _ = plot_kinematics_summary(beta_max=0.95, show=False)
    path = out / "kinematics_summary.png"
    fig.savefig(path, dpi=150)
    print("saved:", path)

    fig, _ = plot_dynamics_summary(mass=1.0, beta_max=0.95, c=1.0, show=False)
    path = out / "dynamics_summary.png"
    fig.savefig(path, dpi=150)
    print("saved:", path)

    fig, _ = plot_decay_summary(proper_lifetime=2.2, velocity=0.95, c=1.0, initial_count=1000, show=False)
    path = out / "decay_summary.png"
    fig.savefig(path, dpi=150)
    print("saved:", path)


if __name__ == "__main__":
    main()
