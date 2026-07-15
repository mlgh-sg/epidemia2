"""Model specification with NumPyro.

The model mirrors the single-population core of the R package: a random walk
(plus intercept) drives the reproduction number ``R_t`` through a link, latent
infections follow the renewal equation, and an observation series is linked to
infections through a delay/ascertainment convolution.

NumPyro is used only to *declare* the model (priors, constraints, likelihood);
inference is performed by BlackJAX (see :mod:`epidemia.infer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .renewal import expected_observations, random_walk, renewal_infections


def link_inv(eta, link):
    """Inverse link mapping the linear predictor to a positive rate."""
    if link == "log":
        return jnp.exp(eta)
    if isinstance(link, tuple) and link[0] == "scaled_logit":
        K = link[1]
        return K / (1.0 + jnp.exp(-eta))
    raise ValueError(f"unknown link: {link!r}")


@dataclass
class EpiConfig:
    """Configuration for a single-population renewal model.

    Attributes
    ----------
    gen : array (L,)
        Generation-interval PMF.
    i2o : array (K,)
        Infection-to-observation delay distribution.
    seed_days : int
        Number of initial days over which infections are seeded.
    link : str | tuple
        ``"log"`` or ``("scaled_logit", K)`` for a carrying-capacity link.
    family : str
        Observation family: ``"poisson"`` or ``"neg_binom"``.
    rw_prior_scale : float
        Scale of the half-normal prior on the random-walk step size.
    intercept_loc, intercept_scale : float
        Normal prior for the R_t intercept (on the link scale).
    seed_prior_mean : float
        Mean of the exponential prior on the (constant) seeded infections.
    rw_index : array (N,) | None
        For each day, the index of the random-walk step it belongs to. Defaults
        to a daily walk (``arange(N)``).
    """

    gen: jnp.ndarray
    i2o: jnp.ndarray
    seed_days: int = 6
    link: object = "log"
    family: str = "poisson"
    rw_prior_scale: float = 0.1
    intercept_loc: float = 0.0
    intercept_scale: float = 0.5
    seed_prior_mean: float = 10.0
    rw_index: object = None
    _extra: dict = field(default_factory=dict, repr=False)


def renewal_model(y, config: EpiConfig):
    """NumPyro model for a single-population renewal process.

    Parameters
    ----------
    y : array (N,)
        Observed series. Use ``nan`` for days that are not observed (e.g. the
        seeding period); those days are masked out of the likelihood.
    config : EpiConfig
        Model configuration.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    N = y.shape[0]

    # the random-walk index is model structure (fixed), so resolve its size
    # statically from NumPy rather than from a traced JAX array.
    rw_index_np = np.arange(N) if config.rw_index is None else np.asarray(config.rw_index)
    n_steps = int(rw_index_np.max()) + 1
    rw_index = jnp.asarray(rw_index_np)

    # --- priors ------------------------------------------------------------
    intercept = numpyro.sample(
        "intercept", dist.Normal(config.intercept_loc, config.intercept_scale)
    )
    rw_scale = numpyro.sample("rw_scale", dist.HalfNormal(config.rw_prior_scale))
    rw_noise = numpyro.sample("rw_noise", dist.Normal(0.0, 1.0).expand([n_steps]))
    seed = numpyro.sample("seed", dist.Exponential(1.0 / config.seed_prior_mean))

    # --- transmission: R_t = g^{-1}(intercept + random walk) ---------------
    walk = random_walk(rw_scale, rw_noise, 0.0)   # cumulative, non-centred
    eta = intercept + walk[rw_index]
    R = link_inv(eta, config.link)
    numpyro.deterministic("Rt", R)

    # --- infections via the renewal equation -------------------------------
    seeds = jnp.full((config.seed_days,), seed)
    infections = renewal_infections(R, seeds, config.gen)
    numpyro.deterministic("infections", infections)

    # --- expected observations ---------------------------------------------
    E = expected_observations(infections, config.i2o, 1.0) + 1e-6
    numpyro.deterministic("E_obs", E)

    # --- likelihood (masking unobserved days) ------------------------------
    mask = ~jnp.isnan(y)
    y_safe = jnp.where(mask, y, 0.0)
    if config.family == "poisson":
        obs_dist = dist.Poisson(E)
    elif config.family == "neg_binom":
        phi = numpyro.sample("reciprocal_dispersion", dist.HalfNormal(5.0))
        obs_dist = dist.NegativeBinomial2(E, phi)
    else:
        raise ValueError(f"unknown family: {config.family!r}")

    with numpyro.handlers.mask(mask=mask):
        numpyro.sample("y", obs_dist, obs=y_safe)
