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
        adaptation="low_rank", backend="numba", progress_bar=True, **kwargs):
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
    progress_bar : bool
        Show nutpie's sampling progress bar (default ``True``). The compile step
        that precedes sampling is announced separately, since it can take longer
        than the sampling itself and has no progress of its own.
    adaptation : {"low_rank", "diag", "flow"}
        nutpie mass-matrix adaptation. Defaults to ``"low_rank"``, which fits a
        low-rank correction to the mass matrix.

        This differs from nutpie's own default of ``"diag"`` deliberately.
        Renewal models put a random walk and a hierarchy into the same posterior,
        which leaves a correlated ridge that a diagonal mass matrix cannot follow
        -- and it fails *silently*, mixing badly without diverging. Measured on
        both a small single-population flu model and the eleven-country
        multilevel one, ``"low_rank"`` was faster **and** better mixed: 4.7x the
        wall-clock speed and 6.3x the effective samples per second on the former,
        1.3x and 4.6x on the latter. There was no case where ``"diag"`` won. See
        the "Performance" page of the documentation.

        ``"flow"`` (normalizing-flow adaptation) helps the hardest posteriors and
        needs ``nutpie[nnflow]``.
    backend : {"numba", "jax"}
        PyTensor compilation backend for the log-density. ``"numba"`` is the
        low-overhead CPU default (recommended, incl. Apple Silicon); ``"jax"``
        is required for GPU (Linux, ``uv sync --extra gpu``) and flow adaptation.

    Returns
    -------
    arviz.InferenceData
        Posterior draws including the ``Rt``/``infections``/``E_obs`` series.
    """
    from .multilevel import _compile

    from .multilevel import _warn_on_divergences

    model = build_model(np.asarray(y), config)
    idata = _compile(model, backend=backend, draws=draws, tune=tune, chains=chains,
                     seed=seed, adaptation=adaptation, progress_bar=progress_bar,
                     **kwargs)
    _warn_on_divergences(idata)
    return idata
