"""
Pytest configuration for the relativity package.

Place this tests/ directory at your project root, next to the relativity/
package folder, then run:

    pytest -q

The path bootstrap below lets the tests run even if the package is not
installed in editable mode yet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Headless-safe plotting tests.
os.environ.setdefault("MPLBACKEND", "Agg")
