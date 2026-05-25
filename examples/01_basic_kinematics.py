"""
Basic special-relativistic kinematics.

Run from project root:
    python examples/01_basic_kinematics.py
"""

from __future__ import annotations

from relativity.sr.kinematics import (
    gamma,
    beta,
    rapidity,
    dilated_time,
    contracted_length,
    lorentz_transform_event_1d,
    inverse_lorentz_transform_event_1d,
    velocity_addition_1d,
    classify_separation,
)

from _helpers import print_header


def main() -> None:
    print_header("01 Basic kinematics")

    c = 1.0
    v = [0.8 * c, 0.0, 0.0]

    print(f"v = {v}")
    print(f"beta = {beta(v, c=c)}")
    print(f"gamma = {gamma(v, c=c)}")
    print(f"rapidity = {rapidity(v, c=c)}")

    tau0 = 2.0
    L0 = 10.0
    print(f"proper time Δτ = {tau0}")
    print(f"lab time Δt = {dilated_time(tau0, v, c=c)}")
    print(f"proper length L0 = {L0}")
    print(f"contracted length L = {contracted_length(L0, v, c=c)}")

    t, x = 2.0, 0.5
    tp, xp = lorentz_transform_event_1d(t, x, 0.5 * c, c=c)
    t_back, x_back = inverse_lorentz_transform_event_1d(tp, xp, 0.5 * c, c=c)

    print("\nLorentz transform 1D")
    print(f"original:  t={t}, x={x}")
    print(f"boosted:   t'={tp}, x'={xp}")
    print(f"recovered: t={t_back}, x={x_back}")

    u = 0.8 * c
    frame_v = 0.5 * c
    u_prime = velocity_addition_1d(u, frame_v, c=c)
    print("\nVelocity addition 1D")
    print(f"u={u}, v={frame_v}, composed={u_prime}, composed/c={u_prime / c}")

    print("\nInterval classification")
    print(classify_separation(0.0, [0, 0, 0], 2.0, [1, 0, 0], c=c))
    print(classify_separation(0.0, [0, 0, 0], 1.0, [1, 0, 0], c=c))
    print(classify_separation(0.0, [0, 0, 0], 1.0, [2, 0, 0], c=c))


if __name__ == "__main__":
    main()
