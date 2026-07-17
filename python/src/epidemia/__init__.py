"""epidemia — Bayesian semi-mechanistic modelling of infectious diseases.

A Python counterpart of the R package `epidemia`. Latent daily infections are
propagated by a renewal process; the reproduction number and observations are
modelled with PyMC and fit with nutpie's fast NUTS.

This Python package is written and maintained by Swapnil Mishra
(https://orcid.org/0000-0002-8759-5902), with Claude Code (Anthropic).

It is a **port, not a new method**. The model, its priors and links, the example
data and the design are from the R package `epidemia` by James A. Scott, Axel
Gandy, Swapnil Mishra, H. Juliette T. Unwin, Seth Flaxman and Samir Bhatt; the
underlying framework is Bhatt et al. (2020), applied in Flaxman et al. (2020).
**Cite those, not this port** -- see the repository's `inst/CITATION`:

* Scott, J. A. et al. (2021). Epidemia: An R Package for Semi-Mechanistic
  Bayesian Modelling of Infectious Diseases using Point Processes.
  arXiv:2110.12461
* Bhatt, S. et al. (2020). Semi-mechanistic Bayesian modelling of COVID-19 with
  renewal processes. arXiv:2012.00394
* Flaxman, S. et al. (2020). Estimating the effects of non-pharmaceutical
  interventions on COVID-19 in Europe. Nature 584, 257-261.

Licensed GPL-3.0-or-later, as the R package is.
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
