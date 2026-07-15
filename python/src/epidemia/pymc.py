"""PyMC model + nutpie sampler backend.

nutpie (a fast Rust NUTS with Fisher-information / low-rank / normalizing-flow
mass-matrix adaptation) front-ends PyMC and Stan — not NumPyro — so this module
provides an *equivalent* renewal model built in PyMC/PyTensor. nutpie's
adaptation is especially helpful for the funnel-prone geometry of random-walk
reproduction-number models.

Requires the optional dependencies::

    uv sync --extra nutpie      # pymc + numba + nutpie
"""

from __future__ import annotations

import numpy as np


def build_pymc_model(y, config):
    """Build the PyMC renewal model. Returns a ``pymc.Model``."""
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    y = np.asarray(y, dtype=float)
    N = y.shape[0]
    gen = np.asarray(config.gen, dtype=float)
    i2o = np.asarray(config.i2o, dtype=float)
    L = gen.shape[0]
    v = int(config.seed_days)
    valid = np.where(np.isfinite(y))[0]
    y_valid = y[valid].astype(int if config.family in ("poisson", "neg_binom") else float)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", config.intercept_loc, config.intercept_scale)
        rw_scale = pm.HalfNormal("rw_scale", config.rw_prior_scale)
        rw_noise = pm.Normal("rw_noise", 0.0, 1.0, shape=N)   # non-centred increments
        seed = pm.Exponential("seed", 1.0 / config.seed_prior_mean)

        # transmission: R_t = link^{-1}(intercept + cumsum(scale * noise))
        eta = intercept + pt.cumsum(rw_scale * rw_noise)
        if config.link == "log":
            R = pt.exp(eta)
        elif isinstance(config.link, tuple) and config.link[0] == "scaled_logit":
            R = config.link[1] * pt.sigmoid(eta)
        else:
            raise ValueError(f"unknown link: {config.link!r}")
        pm.Deterministic("Rt", R)

        # infections via the renewal equation (pytensor scan carrying a window)
        seeds = seed * pt.ones(v)
        rev = seeds[::-1]
        if v >= L:
            buf0 = rev[:L]
        else:
            buf0 = pt.concatenate([rev, pt.zeros(L - v)])

        def step(R_t, buf):
            i_t = R_t * pt.dot(buf, gen)
            new_buf = pt.concatenate([i_t.reshape((1,)), buf[:-1]])
            return new_buf, i_t

        (_, infs), _ = pytensor.scan(
            fn=step, sequences=[R[v:]], outputs_info=[buf0, None]
        )
        infections = pt.concatenate([seeds, infs])
        pm.Deterministic("infections", infections)

        # expected observations: causal convolution with the i2o delay
        terms = [i2o[0] * infections]
        for k in range(1, len(i2o)):
            terms.append(i2o[k] * pt.concatenate([pt.zeros(k), infections[:-k]]))
        E = pt.add(*terms) + 1e-6 if len(terms) > 1 else terms[0] + 1e-6
        pm.Deterministic("E_obs", E)

        # likelihood on observed days only
        mu = E[valid]
        if config.family == "poisson":
            pm.Poisson("y", mu=mu, observed=y_valid)
        elif config.family == "neg_binom":
            phi = pm.HalfNormal("reciprocal_dispersion", 5.0)
            pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=y_valid)
        else:
            raise ValueError(f"unknown family: {config.family!r}")

    return model


def fit_nutpie(y, config, draws=1000, tune=1000, chains=4, seed=0,
               adaptation="diag", backend="numba", **kwargs):
    """Fit the PyMC renewal model with nutpie's Rust NUTS.

    Parameters
    ----------
    adaptation : {"diag", "low_rank", "flow"}
        nutpie mass-matrix adaptation. ``"diag"`` (Fisher-information diagonal)
        is a strong default; ``"low_rank"`` and ``"flow"`` help hard posteriors.
    backend : {"numba", "jax"}
        PyTensor compilation backend for the log-density. ``"numba"`` is the
        low-overhead CPU default; ``"jax"`` is required for GPU / flow adaptation.
    """
    import nutpie

    model = build_pymc_model(y, config)
    compiled = nutpie.compile_pymc_model(model, backend=backend)
    idata = nutpie.sample(
        compiled, draws=draws, tune=tune, chains=chains, seed=seed,
        adaptation=adaptation, progress_bar=kwargs.get("progress_bar", False),
    )
    return idata
