"""epidemia — Bayesian semi-mechanistic modelling of infectious diseases.

A Python counterpart of the R package `epidemia`. Latent daily infections are
propagated by a renewal process; the reproduction number and observations are
modelled with NumPyro, and inference uses a fast NUTS backend (nutpie by
default, or BlackJAX / NumPyro).
"""

from __future__ import annotations

# 64-bit precision keeps the renewal recursion and NUTS numerically stable.
# Must run before JAX creates any arrays; this package is imported first.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

from . import plots  # noqa: E402
from .data import EpiData, flu1918  # noqa: E402
from .infer import fit  # noqa: E402
from .model import EpiConfig, link_inv, renewal_model  # noqa: E402
from .renewal import (  # noqa: E402
    expected_observations,
    infectiousness,
    random_walk,
    renewal_infections,
)

__all__ = [
    "EpiConfig",
    "EpiData",
    "expected_observations",
    "fit",
    "flu1918",
    "infectiousness",
    "link_inv",
    "plots",
    "random_walk",
    "renewal_infections",
    "renewal_model",
]
__version__ = "0.1.0"
