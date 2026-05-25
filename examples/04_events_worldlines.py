"""
Events and worldlines.

Run from project root:
    python examples/04_events_worldlines.py
"""

from __future__ import annotations

from relativity.physics.event import Event
from relativity.physics.worldline import Worldline
from relativity.sr.kinematics import interval_between_events, proper_time_between_events

from _helpers import print_header


def main() -> None:
    print_header("04 Events and worldlines")

    c = 1.0

    A = Event(0.0, [0.0, 0.0, 0.0], c=c)
    B = Event(1.0, [0.5, 0.0, 0.0], c=c)
    C = Event(2.0, [1.0, 0.0, 0.0], c=c)

    print("A:", A)
    print("B:", B)
    print("C:", C)

    print("\nInterval A->B:", interval_between_events(A.t, A.r, B.t, B.r, c=c))
    print("Proper time A->B:", proper_time_between_events(A.t, A.r, B.t, B.r, c=c))

    wl = Worldline([A, B, C], c=c)

    print("\nWorldline:", wl)
    print("times:", wl.times)
    print("positions:", wl.positions)
    print("segment velocities:", wl.velocities())
    print("segment gamma factors:", wl.gamma_factors())
    print("total proper time:", wl.proper_time())

    print("\nExpected rejection of spacelike worldline segment:")
    try:
        bad = Worldline(
            [Event(0.0, [0.0, 0.0, 0.0], c=c), Event(1.0, [2.0, 0.0, 0.0], c=c)],
            c=c,
        )
        bad.proper_time()
    except ValueError as exc:
        print("caught:", exc)


if __name__ == "__main__":
    main()
