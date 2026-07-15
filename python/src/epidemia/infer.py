"""Inference with nutpie.

The PyMC renewal model is sampled with **nutpie**, a fast Rust NUTS with
efficient (Fisher-information / low-rank / normalizing-flow) mass-matrix
adaptation. It returns an ArviZ ``InferenceData`` whose posterior contains the
sampled parameters plus the latent series ``Rt``, ``infections`` and ``E_obs``.
"""

from __future__ import annotations

import numpy as np

from .model import build_model


def fit(y, config, draws=1000, tune=1000, chains=4, seed=0,
        adaptation="diag", backend="numba", progress_bar=False, **kwargs):
    """Fit a single-population renewal model with nutpie.

    Parameters
    ----------
    y : array (N,)
        Observed series (``nan`` for unobserved/seeding days).
    config : epidemia.model.EpiConfig
        Model configuration.
    draws, tune, chains : int
        Posterior draws per chain, warmup/tuning iterations, and number of chains.
    seed : int
        Random seed.
    adaptation : {"diag", "low_rank", "flow"}
        nutpie mass-matrix adaptation. ``"diag"`` (a Fisher-information diagonal)
        is a strong default; ``"low_rank"`` and ``"flow"`` help hard posteriors
        (``"flow"`` needs ``nutpie[nnflow]``).
    backend : {"numba", "jax"}
        PyTensor compilation backend for the log-density. ``"numba"`` is the
        low-overhead CPU default (recommended, incl. Apple Silicon); ``"jax"``
        is required for GPU (Linux, ``uv sync --extra gpu``) and flow adaptation.

    Returns
    -------
    arviz.InferenceData
        Posterior draws including the ``Rt``/``infections``/``E_obs`` series.
    """
    import nutpie

    model = build_model(np.asarray(y), config)
    kw = {}
    if backend == "jax":
        kw["gradient_backend"] = "jax"
    compiled = nutpie.compile_pymc_model(model, backend=backend, **kw)
    idata = nutpie.sample(
        compiled, draws=draws, tune=tune, chains=chains, seed=seed,
        adaptation=adaptation, progress_bar=progress_bar, **kwargs,
    )
    return idata
