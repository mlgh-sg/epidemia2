"""Variational inference (ADVI), mirroring R's ``epim(algorithm = "meanfield"/"fullrank")``.

The R package fits with Stan's ADVI when ``algorithm`` is ``"meanfield"`` or
``"fullrank"``; the Python port offers the same two approximating families
through PyMC's ADVI implementations::

    "meanfield"  ->  pm.fit(method="advi")           # diagonal Gaussian
    "fullrank"   ->  pm.fit(method="fullrank_advi")  # Gaussian with full covariance

.. warning::

   **Variational Bayes understates posterior uncertainty and must not be used
   for final inference.** ADVI fits the *closest* Gaussian (in reverse KL) to
   the posterior, and reverse KL is mode-seeking: credible intervals come out
   too narrow, correlations are lost entirely under ``"meanfield"``, and there
   is no diagnostic comparable to R-hat that tells you the approximation is
   wrong. The R documentation says the same thing. Use it to iterate quickly on
   model structure and to get starting values; report NUTS results
   (:func:`epidemia.fit`, :func:`epidemia.fit_multilevel`).

ADVI also stops when it runs out of iterations rather than when it has
converged -- R's multilevel vignette hits exactly this and Stan prints *"the
maximum number of iterations is reached"*. :func:`fit_variational` therefore
checks whether the ELBO has plateaued and warns if it is still climbing, and
returns the whole ELBO trace in the ``elbo`` group of the ``InferenceData`` so
you can look at it yourself.
"""

from __future__ import annotations

import warnings

import numpy as np

# R's algorithm names -> PyMC's `method` strings. The PyMC spellings are
# accepted too, so code written against either package's vocabulary works.
_ALGORITHMS = {
    "meanfield": "advi",
    "fullrank": "fullrank_advi",
    "advi": "advi",
    "fullrank_advi": "fullrank_advi",
}

#: Default relative-improvement tolerance for the ELBO plateau check. Matches
#: Stan's ``tol_rel_obj`` (0.01), which is what R's ADVI fits are judged by.
DEFAULT_RTOL = 0.01



class _ElboConvergence:
    """Stop once the ELBO's relative improvement stays below ``rtol``.

    R forwards ``tol_rel_obj`` to CmdStan, which watches the **ELBO** -- not the
    variational parameters, which is what PyMC's own
    ``CheckParametersConvergence`` tracks. Using the parameter-based callback
    stops at a different point and disagrees with
    :func:`elbo_relative_improvement`, the measure this module reports
    afterwards, so a run could stop early and still be labelled unconverged.

    The ELBO is a stochastic estimate, so a single quiet stretch is not
    convergence: two full blocks of it are required, and no check happens until
    ``min_iter``. Stopping on the first quiet check ends the fit while the
    optimiser is still climbing -- on the test model that left the posterior
    mean at 0.19 instead of 3.26.
    """

    def __init__(self, rtol, every=500, min_iter=5000, consecutive=2):
        self.rtol = float(rtol)
        self.every = int(every)
        self.min_iter = int(min_iter)
        self.consecutive = int(consecutive)
        self._hits = 0

    def __call__(self, approx, loss, i):
        if i < self.min_iter or i % self.every:
            return
        elbo = -np.asarray(loss[: i + 1], dtype=float)
        rel = elbo_relative_improvement(elbo)
        if np.isfinite(rel) and rel < self.rtol:
            self._hits += 1
            if self._hits >= self.consecutive:
                raise StopIteration(f"ELBO converged at iteration {i}")
        else:
            self._hits = 0


def fit_variational(model, algorithm="fullrank", iter=50000, draws=1000, seed=0,
                    progress_bar=True, rtol=DEFAULT_RTOL, early_stop=True,
                    **kwargs):
    """Fit a PyMC model by automatic differentiation variational inference.

    Parameters
    ----------
    model : pymc.Model
        The model to fit -- e.g. the output of :func:`epidemia.build_model` or
        :func:`epidemia.build_multilevel_model`.
    algorithm : {"meanfield", "fullrank"}
        Approximating family, named as in R's ``epim``. ``"meanfield"`` is a
        diagonal Gaussian (fast, ignores all posterior correlation);
        ``"fullrank"`` estimates a full covariance (slower, still Gaussian).
        The PyMC names ``"advi"``/``"fullrank_advi"`` are accepted as synonyms.
    iter : int
        Maximum number of stochastic-gradient iterations. As in Stan, the
        optimiser simply stops here -- reaching ``iter`` is not evidence of
        convergence, which is why the ELBO is checked afterwards.
    draws : int
        Number of samples drawn from the fitted approximation.
    seed : int
        Random seed, used for both the optimisation and the draws (offset by
        one, so the draws are not the optimiser's own noise stream).
    progress_bar : bool
        Show PyMC's ADVI progress bar (which reports the running average loss).
    early_stop : bool, default True
        Stop once the ELBO's relative change falls below ``rtol``, which is what
        R's ``tol_rel_obj`` does. ``False`` always runs the full ``iter``
        budget. Passing your own ``callbacks=`` overrides this either way.
    rtol : float
        Relative-improvement tolerance for the plateau check; see
        :func:`elbo_relative_improvement`.
    **kwargs
        Passed to :func:`pymc.fit` (e.g. ``obj_optimizer=pm.adam(...)``,
        ``callbacks=[...]``, ``obj_n_mc=...``).

    Returns
    -------
    arviz.InferenceData
        Draws from the approximation, shaped exactly like a NUTS fit (one chain
        of ``draws`` draws) so downstream code -- plotting, forecasting,
        ``az.summary`` -- works unchanged. Deterministics (``Rt``,
        ``infections``, ``E_obs``, ...) are included. Two extras are attached:

        ``idata.elbo``
            Group holding the ELBO trace, ``elbo["elbo"]`` over an
            ``iteration`` dimension (higher is better; it is the negative of
            PyMC's loss history).
        ``idata.attrs``
            ``inference_method``, ``algorithm``, ``iter``, ``elbo``,
            ``elbo_rel_improvement`` and ``elbo_converged``.

    Warns
    -----
    UserWarning
        Always, that variational estimates of uncertainty are optimistic; and
        additionally if the ELBO has not plateaued by iteration ``iter``.

    Notes
    -----
    **Do not report variational intervals.** See the module docstring: this is a
    development-loop tool, not an inference tool.

    Examples
    --------
    >>> import epidemia as epi                                # doctest: +SKIP
    >>> model = epi.build_model(y, config)                    # doctest: +SKIP
    >>> idata = epi.fit_variational(model, "meanfield", iter=20000)  # doctest: +SKIP
    >>> idata.elbo["elbo"].values[-5:]                        # doctest: +SKIP
    """
    import pymc as pm

    method = _resolve_algorithm(algorithm)

    warnings.warn(
        "Variational inference (ADVI) understates posterior uncertainty: "
        "intervals are too narrow and, under algorithm='meanfield', posterior "
        "correlations are dropped altogether. Use it to iterate on the model, "
        "then refit with NUTS (epidemia.fit / epidemia.fit_multilevel) for any "
        "result you intend to report.",
        stacklevel=2,
    )

    # R forwards tol_rel_obj to CmdStan, which STOPS the optimiser once the
    # ELBO's relative change falls below it (R/backend.R:67; CmdStan defaults to
    # 0.01). Without a callback PyMC always burns the full budget, so an R fit
    # that converges in 800 iterations took `iter` iterations here.
    callbacks = kwargs.pop("callbacks", None)
    if callbacks is None and early_stop:
        callbacks = [_ElboConvergence(rtol)]

    with model:
        approx = pm.fit(n=int(iter), method=method, random_seed=seed,
                        progressbar=progress_bar, callbacks=callbacks, **kwargs)

    # PyMC minimises the loss (= -ELBO); flip it so "up" means "better", which
    # is how the ELBO is reported everywhere else (Stan, the ADVI paper, R).
    elbo = -np.asarray(approx.hist, dtype=float)

    idata = approx.sample(draws=int(draws), random_seed=seed + 1)
    _attach_elbo(idata, elbo, algorithm=algorithm, method=method, iter=int(iter),
                 rtol=rtol)
    _warn_if_not_plateaued(elbo, iter=int(iter), rtol=rtol)
    return idata


def _resolve_algorithm(algorithm) -> str:
    """Map an R-style ``algorithm`` name onto a PyMC ``method`` string."""
    try:
        return _ALGORITHMS[algorithm]
    except (KeyError, TypeError):
        valid = ", ".join(repr(k) for k in ("meanfield", "fullrank"))
        raise ValueError(
            f"unknown variational algorithm {algorithm!r}; valid values are "
            f"{valid} (as in R's epim), or the PyMC synonyms 'advi' and "
            "'fullrank_advi'. For NUTS/MCMC use epidemia.fit or "
            "epidemia.fit_multilevel instead."
        ) from None


def elbo_relative_improvement(elbo, frac: float = 0.1) -> float:
    """Relative gain in the ELBO over the final ``frac`` of the optimisation.

    Compares the mean ELBO of the last ``frac`` of the history against the mean
    over the ``frac`` before it::

        (mean(last block) - mean(previous block)) / |mean(previous block)|

    Parameters
    ----------
    elbo : array (n_iter,)
        ELBO trace, higher-is-better (i.e. the negative of PyMC's loss history).
    frac : float
        Fraction of the history in each block (default 0.1, i.e. last 10% vs the
        preceding 10%).

    Returns
    -------
    float
        Positive when the ELBO is still climbing (not converged), ~0 on a
        plateau. ``nan`` if the history is too short to split into two blocks.

    Notes
    -----
    The ELBO is a *stochastic* estimate, so this is noisy -- especially for
    ``"fullrank"``, whose gradient variance is much higher. A single value just
    below the tolerance is not proof of convergence; plot the trace.
    """
    elbo = np.asarray(elbo, dtype=float)
    n = elbo.shape[0]
    k = max(1, int(round(n * frac)))
    if n < 2 * k:
        return float("nan")
    last = float(elbo[-k:].mean())
    prev = float(elbo[-2 * k:-k].mean())
    # Guard the scale: early ELBOs can pass through ~0 and the ratio would blow
    # up. Dividing by the larger magnitude keeps the statistic in [-2, 2].
    scale = max(abs(prev), abs(last), 1e-12)
    return (last - prev) / scale


def _warn_if_not_plateaued(elbo, iter: int, rtol: float) -> None:
    """Warn when the optimiser ran out of iterations instead of converging."""
    rel = elbo_relative_improvement(elbo)
    if np.isnan(rel) or rel <= rtol:
        return
    warnings.warn(
        f"the maximum number of iterations is reached: the ELBO was still "
        f"improving by {rel:.3%} over the last 10% of the {iter} iterations "
        f"(tolerance {rtol:.3%}), so the approximation has not converged. "
        "Increase `iter`, or pass a different `obj_optimizer`. Inspect the "
        "trace with idata.elbo['elbo'].",
        stacklevel=3,
    )


def _attach_elbo(idata, elbo, algorithm: str, method: str, iter: int,
                 rtol: float) -> None:
    """Attach the ELBO trace (as a group) and the convergence summary (attrs)."""
    import xarray as xr

    rel = elbo_relative_improvement(elbo)
    idata.add_groups(
        {"elbo": xr.Dataset({"elbo": ("iteration", elbo)})},
        warn_on_custom_groups=False,
    )
    idata.attrs.update({
        "inference_method": "variational",
        "algorithm": algorithm,
        "method": method,
        "iter": iter,
        "elbo": float(elbo[-1]) if elbo.size else float("nan"),
        "elbo_rel_improvement": rel,
        # np.nan -> False: an unknowable convergence status is not convergence.
        "elbo_converged": bool(not np.isnan(rel) and rel <= rtol),
    })
