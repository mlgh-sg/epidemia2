"""Model specification with PyMC.

The model mirrors the single-population core of the R package: a (non-centred)
random walk plus intercept drives the reproduction number ``R_t`` through a
link, latent infections follow the renewal equation, and an observation series
is linked to infections through a delay/ascertainment convolution.

PyMC declares the model (priors, the renewal recursion via ``pytensor.scan``,
the likelihood); inference is done by nutpie (see :mod:`epidemia.infer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpiConfig:
    """Configuration for a single-population renewal model.

    Attributes
    ----------
    gen : array (L,)
        Generation-interval kernel; ``gen[k]`` weights the infection ``k+1``
        days in the past (drop the same-day serial-interval entry).
    i2o : array (K,)
        Infection-to-observation delay distribution; ``i2o[k]`` weights
        infections ``k`` days before the observation.
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
        to a daily walk (``arange(N)``); pass e.g. a week index for a weekly walk.
    """

    gen: np.ndarray
    i2o: np.ndarray
    seed_days: int = 6
    link: object = "log"
    family: str = "poisson"
    rw_prior_scale: float = 0.1
    intercept_loc: float = 0.0
    intercept_scale: float = 0.5
    seed_prior_mean: float = 10.0
    rw_index: object = None
    _extra: dict = field(default_factory=dict, repr=False)


def build_model(y, config: EpiConfig):
    """Build the PyMC renewal model. Returns a :class:`pymc.Model`.

    Parameters
    ----------
    y : array (N,)
        Observed series. Use ``nan`` for days that are not observed (e.g. the
        seeding period); those days are dropped from the likelihood.
    config : EpiConfig
        Model configuration.
    """
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    y = np.asarray(y, dtype=float)
    N = y.shape[0]
    gen = np.asarray(config.gen, dtype=float)
    i2o = np.asarray(config.i2o, dtype=float)
    L = gen.shape[0]
    v = int(config.seed_days)

    rw_index = np.arange(N) if config.rw_index is None else np.asarray(config.rw_index)
    n_steps = int(rw_index.max()) + 1

    valid = np.where(np.isfinite(y))[0]
    is_count = config.family in ("poisson", "neg_binom")
    y_valid = y[valid].astype(int if is_count else float)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", config.intercept_loc, config.intercept_scale)
        rw_scale = pm.HalfNormal("rw_scale", config.rw_prior_scale)
        rw_noise = pm.Normal("rw_noise", 0.0, 1.0, shape=n_steps)  # non-centred increments
        seed = pm.Exponential("seed", 1.0 / config.seed_prior_mean)

        # transmission: R_t = link^{-1}(intercept + cumsum(scale * noise))
        walk = pt.cumsum(rw_scale * rw_noise)
        eta = intercept + walk[rw_index]
        if config.link == "log":
            R = pt.exp(eta)
        elif isinstance(config.link, tuple) and config.link[0] == "scaled_logit":
            R = config.link[1] * pt.sigmoid(eta)
        else:
            raise ValueError(f"unknown link: {config.link!r}")
        pm.Deterministic("Rt", R)

        # Infections via the renewal equation. This is a *time-varying* linear
        # recursion (R_t varies and the output feeds back), so it has no
        # convolution/FFT form and must stay a sequential scan -- carrying a
        # length-L window and taking one dot product (the renewal weight) per
        # step, which is the minimal per-step work.
        seeds = seed * pt.ones(v)
        rev = seeds[::-1]
        buf0 = rev[:L] if v >= L else pt.concatenate([rev, pt.zeros(L - v)])

        def step(R_t, buf):
            i_t = R_t * pt.dot(buf, gen)
            new_buf = pt.concatenate([i_t.reshape((1,)), buf[:-1]])
            return new_buf, i_t

        _, infs = pytensor.scan(
            fn=step, sequences=[R[v:]], outputs_info=[buf0, None], return_updates=False,
        )
        infections = pt.concatenate([seeds, infs])
        pm.Deterministic("infections", infections)

        # Expected observations: a *time-invariant* delay, so (unlike the renewal
        # above) it IS a convolution -- computed vectorised as a sum of K shifted
        # infection series, with no scan. The `for k` unrolls the K-tap kernel at
        # graph-build time; it is not a runtime loop.
        terms = [i2o[0] * infections]
        for k in range(1, len(i2o)):
            terms.append(i2o[k] * pt.concatenate([pt.zeros(k), infections[:-k]]))
        E = (pt.add(*terms) if len(terms) > 1 else terms[0]) + 1e-6
        pm.Deterministic("E_obs", E)

        # likelihood on observed days only
        mu = E[valid]
        if config.family == "poisson":
            pm.Poisson("y", mu=mu, observed=y_valid)
        elif config.family == "neg_binom":
            phi = pm.HalfNormal("reciprocal_dispersion", 5.0)
            pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=y_valid)
        elif config.family == "normal":
            sigma = pm.HalfNormal("obs_sigma", 5.0)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_valid)
        else:
            raise ValueError(f"unknown family: {config.family!r}")

    return model
