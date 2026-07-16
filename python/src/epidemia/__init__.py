"""epidemia — Bayesian semi-mechanistic modelling of infectious diseases.

A Python counterpart of the R package `epidemia`. Latent daily infections are
propagated by a renewal process; the reproduction number and observations are
modelled with PyMC and fit with nutpie's fast NUTS.
"""

from __future__ import annotations

from . import plots
from .data import EUROPE_COVID_NPIS, EpiData, EuropeCovid2, europe_covid2, flu1918
from .infer import fit
from .model import EpiConfig, build_model
from .multilevel import (
    MultilevelConfig,
    MultilevelData,
    build_multilevel_model,
    effect_table,
    fit_multilevel,
    prepare_panel,
)
from .renewal import (
    expected_observations,
    infectiousness,
    random_walk,
    renewal_infections,
)

__all__ = [
    "EUROPE_COVID_NPIS",
    "EpiConfig",
    "EpiData",
    "EuropeCovid2",
    "MultilevelConfig",
    "MultilevelData",
    "build_model",
    "build_multilevel_model",
    "effect_table",
    "europe_covid2",
    "fit_multilevel",
    "prepare_panel",
    "expected_observations",
    "fit",
    "flu1918",
    "infectiousness",
    "plots",
    "random_walk",
    "renewal_infections",
]
__version__ = "0.1.0"
