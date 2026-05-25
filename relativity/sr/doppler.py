"""Relativistic Doppler and aberration formulas."""
from __future__ import annotations
from relativity.utils import smart_sqrt


def longitudinal_doppler_frequency(f_source, beta, approaching=False):
    """Observed frequency for longitudinal motion. beta>0 means receding unless approaching=True."""
    if approaching:
        return f_source * smart_sqrt((1 + beta) / (1 - beta))
    return f_source * smart_sqrt((1 - beta) / (1 + beta))


def doppler_factor(beta, cos_theta=1):
    """D = sqrt(1-beta^2)/(1-beta cos(theta)); f_obs = D f_emit."""
    return smart_sqrt(1 - beta**2) / (1 - beta * cos_theta)


def observed_frequency(f_emit, beta, cos_theta=1):
    return doppler_factor(beta, cos_theta) * f_emit


def redshift_from_frequencies(f_emit, f_obs):
    return f_emit / f_obs - 1


__all__ = ["longitudinal_doppler_frequency", "doppler_factor", "observed_frequency", "redshift_from_frequencies"]
