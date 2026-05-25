"""Small helpers shared by examples."""

from __future__ import annotations

from pathlib import Path


def output_dir() -> Path:
    """Return examples/output, creating it if needed."""
    path = Path(__file__).resolve().parent / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")
