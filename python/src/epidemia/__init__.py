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

from . import plots, predict, priors, scoring
from .core import (
    EpiModelConfig,
    ObsModel,
    PanelData,
    RandomWalk,
    build_epidemia_model,
)
from .core import prepare_panel as prepare_panel_multi
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
from .plots import spaghetti_infections, spaghetti_obs, spaghetti_rt
from .predict import posterior_predict, simulate
from .renewal import (
    expected_observations,
    infectiousness,
    random_walk,
    renewal_infections,
)
from .scoring import crps, evaluate_forecast, posterior_coverage, posterior_metrics
from .variational import fit_variational

__all__ = [
    "EUROPE_COVID_NPIS",
    "EpiConfig",
    "EpiData",
    "EpiModelConfig",
    "EuropeCovid2",
    "MultilevelConfig",
    "MultilevelData",
    "ObsModel",
    "PanelData",
    "RandomWalk",
    "build_epidemia_model",
    "build_model",
    "build_multilevel_model",
    "crps",
    "effect_table",
    "europe_covid2",
    "evaluate_forecast",
    "expected_observations",
    "fit",
    "fit_multilevel",
    "fit_variational",
    "flu1918",
    "infectiousness",
    "plots",
    "posterior_coverage",
    "posterior_metrics",
    "posterior_predict",
    "predict",
    "prepare_panel",
    "prepare_panel_multi",
    "priors",
    "random_walk",
    "renewal_infections",
    "scoring",
    "simulate",
    "spaghetti_infections",
    "spaghetti_obs",
    "spaghetti_rt",
]

# Two names appear in more than one module, so the top-level binding picks one
# and the other stays reachable through its module:
#   expected_observations -> renewal's single-series NumPy reference.
#       predict.expected_observations is the draw-batched version.
#   prepare_panel -> multilevel's single-response version, kept for the existing
#       notebooks. core.prepare_panel, re-exported as prepare_panel_multi,
#       handles several response series at once.
__version__ = "0.1.0"
