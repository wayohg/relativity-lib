"""
Doppler shift and relativistic decay examples.

Run from project root:
    python examples/06_doppler_decay.py
"""

from __future__ import annotations

from relativity.sr.doppler import (
    longitudinal_doppler_factor,
    observed_frequency,
    redshift_from_beta,
    beta_from_longitudinal_redshift,
)
from relativity.sr.decay import (
    lab_lifetime,
    decay_length,
    survival_probability_time,
    survival_probability_distance,
    two_body_decay_momentum,
    two_body_decay_energies,
    q_value,
)

from _helpers import print_header


def main() -> None:
    print_header("06 Doppler and decay")

    beta = 0.7
    f_emit = 1.0e9

    print("Doppler")
    D_rec = longitudinal_doppler_factor(beta, approaching=False)
    D_app = longitudinal_doppler_factor(beta, approaching=True)
    print("receding factor:", D_rec)
    print("approaching factor:", D_app)
    print("observed receding frequency:", observed_frequency(f_emit, doppler=D_rec))
    print("observed approaching frequency:", observed_frequency(f_emit, doppler=D_app))

    z = redshift_from_beta(beta)
    print("redshift z:", z)
    print("beta recovered from z:", beta_from_longitudinal_redshift(z))

    print("\nDecay")
    c = 1.0
    tau0 = 2.2
    v = 0.95 * c
    print("proper lifetime:", tau0)
    print("lab lifetime:", lab_lifetime(tau0, v, c=c))
    print("mean decay length:", decay_length(tau0, v, c=c))
    print("survival at t=10:", survival_probability_time(10.0, tau0, v, c=c))
    print("survival at x=10:", survival_probability_distance(10.0, tau0, v, c=c))

    print("\nTwo-body decay")
    M, m1, m2 = 5.0, 1.0, 2.0
    print("Q-value:", q_value(M, [m1, m2], c=c))
    print("daughter momentum magnitude:", two_body_decay_momentum(M, m1, m2, c=c))
    print("daughter energies:", two_body_decay_energies(M, m1, m2, c=c))


if __name__ == "__main__":
    main()
