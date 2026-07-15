"""Inference: fit a renewal model and return an ArviZ ``InferenceData``.

Three sampler backends are available, all fitting the same NumPyro model:

* ``"nutpie"`` (default) — a fast Rust NUTS with efficient mass-matrix
  adaptation; easy to install (pre-built wheels) and typically the quickest.
* ``"numpyro"`` — NumPyro's own NUTS (captures deterministic sites natively).
* ``"blackjax"`` — BlackJAX NUTS with window adaptation, chains via ``vmap``.

Whatever the backend, the returned ``InferenceData`` posterior contains the
sampled parameters plus the latent series ``Rt``, ``infections`` and ``E_obs``.
"""

from __future__ import annotations

from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from .model import renewal_model

_DETERMINISTICS = ("Rt", "infections", "E_obs")


def fit(y, config, sampler="numpyro", draws=1000, tune=1000, chains=4, seed=0,
        init_scale=0.1, **kwargs):
    """Fit a single-population renewal model.

    Parameters
    ----------
    y : array (N,)
        Observed series (``nan`` for unobserved/seeding days).
    config : epidemia.model.EpiConfig
        Model configuration.
    sampler : {"numpyro", "blackjax", "nutpie"}
        Inference backend. ``"numpyro"`` (default) uses NumPyro's NUTS, which is
        GPU-capable and captures the latent series natively. ``"nutpie"`` uses
        the fast Rust NUTS via an equivalent PyMC model (see :mod:`epidemia.pymc`)
        and requires ``pymc`` + ``nutpie`` (``uv sync --extra nutpie``).
    draws, tune, chains : int
        Posterior draws per chain, warmup/tuning iterations, and number of chains.
    seed : int
        Random seed.

    Returns
    -------
    arviz.InferenceData
        Posterior draws including the ``Rt``/``infections``/``E_obs`` series.
    """
    y = jnp.asarray(y)

    def model():
        return renewal_model(y, config)

    if sampler == "numpyro":
        idata = _fit_numpyro(model, draws, tune, chains, seed)
    elif sampler == "blackjax":
        idata = _fit_blackjax(model, draws, tune, chains, seed, init_scale)
        idata = _augment_with_deterministics(idata, model, seed)
    elif sampler == "nutpie":
        # nutpie front-ends PyMC/Stan (not NumPyro), so route to the PyMC model.
        from .pymc import fit_nutpie
        idata = fit_nutpie(np.asarray(y), config, draws=draws, tune=tune,
                           chains=chains, seed=seed, **kwargs)
    else:
        raise ValueError(f"unknown sampler: {sampler!r}")
    return idata


# --------------------------------------------------------------------------- #
# NumPyro NUTS (captures deterministics natively)
# --------------------------------------------------------------------------- #
def _fit_numpyro(model, draws, tune, chains, seed):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=tune, num_samples=draws, num_chains=chains,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return az.from_numpyro(mcmc)


# --------------------------------------------------------------------------- #
# BlackJAX NUTS
# --------------------------------------------------------------------------- #
def _fit_blackjax(model, draws, tune, chains, seed, init_scale):
    import blackjax
    from numpyro.infer.util import initialize_model

    key = jax.random.PRNGKey(seed)
    init_key, warm_key, sample_key = jax.random.split(key, 3)

    model_info = initialize_model(init_key, model, dynamic_args=False)
    logdensity = lambda p: -model_info.potential_fn(p)
    z0 = model_info.param_info.z  # unconstrained init

    def run_chain(chain_key, jitter_key):
        # over-dispersed init per chain
        z_init = jax.tree_util.tree_map(
            lambda v, k: v + init_scale * jax.random.normal(k, v.shape),
            z0, _split_like(jitter_key, z0),
        )
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, params), _ = warmup.run(chain_key, z_init, num_steps=tune)
        kernel = blackjax.nuts(logdensity, **params)

        def step(state, k):
            state, info = kernel.step(k, state)
            return state, (state.position, info.is_divergent)

        keys = jax.random.split(chain_key, draws)
        _, (positions, divs) = jax.lax.scan(step, state, keys)
        return positions, divs

    ckeys = jax.random.split(warm_key, chains)
    jkeys = jax.random.split(sample_key, chains)
    positions, divs = jax.vmap(run_chain)(ckeys, jkeys)

    # transform unconstrained -> constrained and to arviz
    constrained = jax.vmap(jax.vmap(model_info.postprocess_fn))(positions)
    post = {k: np.asarray(v) for k, v in constrained.items()}
    return az.from_dict(posterior=post,
                        sample_stats={"diverging": np.asarray(divs)})


def _split_like(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(treedef, keys)


# --------------------------------------------------------------------------- #
# recover deterministic sites (Rt, infections, E_obs) via Predictive
# --------------------------------------------------------------------------- #
def _augment_with_deterministics(idata, model, seed):
    """Add posterior draws of the deterministic time-series to ``idata``."""
    post = idata.posterior
    sampled = [v for v in post.data_vars if v not in _DETERMINISTICS]
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]

    # flatten (chain, draw, ...) -> (chain*draw, ...) for Predictive
    flat = {}
    for v in sampled:
        arr = np.asarray(post[v])
        flat[v] = arr.reshape(n_chain * n_draw, *arr.shape[2:])

    pred = Predictive(model, posterior_samples={k: jnp.asarray(v) for k, v in flat.items()},
                      return_sites=_DETERMINISTICS)
    det = pred(jax.random.PRNGKey(seed))

    import xarray as xr
    for name in _DETERMINISTICS:
        if name not in det:
            continue
        arr = np.asarray(det[name]).reshape(n_chain, n_draw, -1)
        dim = f"{name}_dim"
        post[name] = xr.DataArray(arr, dims=("chain", "draw", dim))
    return idata
