"""epidemia — Bayesian semi-mechanistic modelling of infectious diseases.

A Python counterpart of the R package `epidemia`. Latent daily infections are
propagated by a renewal process; the reproduction number and observations are
modelled with PyMC and fit with nutpie's fast NUTS.
"""

from __future__ import annotations

from . import plots
from .data import EpiData, flu1918
from .infer import fit
from .model import EpiConfig, build_model
from .renewal import (
    expected_observations,
    infectiousness,
    random_walk,
    renewal_infections,
)

__all__ = [
    "EpiConfig",
    "EpiData",
    "build_model",
    "expected_observations",
    "fit",
    "flu1918",
    "infectiousness",
    "plots",
    "random_walk",
    "renewal_infections",
]
__version__ = "0.1.0"
