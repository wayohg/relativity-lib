"""
Physical constants for relativistic calculations.
All values in SI units unless noted.
"""

C = 299_792_458.0          # Speed of light (m/s)
PLANCK = 6.62607015e-34    # Planck's constant (J·s)
ELECTRON_VOLT = 1.602176634e-19  # 1 eV in Joules

KEV = 1e3  * ELECTRON_VOLT
MEV = 1e6  * ELECTRON_VOLT
GEV = 1e9  * ELECTRON_VOLT
TEV = 1e12 * ELECTRON_VOLT

YEAR      = 365.25 * 24 * 3600     # seconds
LIGHTYEAR = C * YEAR               # metres
