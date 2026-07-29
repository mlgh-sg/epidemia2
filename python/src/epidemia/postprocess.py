"""Post-processing helpers, mirroring R's ``posterior_*`` family.

R exposes several ways to interrogate a fit that the port had no counterpart
for: decomposing the linear predictor, recovering the total infectiousness
series, reporting the priors actually in force, and pulling draws out by name.
Everything here operates on an ArviZ ``InferenceData`` plus the objects that
produced it, and none of it refits.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "posterior_linpred",
    "posterior_infectious",
    "extract_samples",
    "prior_summary",
]


def _flat(da):
    """(chain, draw, ...) -> (draws, ...)."""
    a = np.asarray(da)
    return a.reshape((-1,) + a.shape[2:])


def posterior_linpred(idata, panel, config, series=None, obs_models=None,
                      transform=False, fixed=True, random=True, autocor=True):
    """Draws of a linear predictor, with components switchable.

    Mirrors R's ``posterior_linpred``. ``series=None`` gives the predictor for
    ``R_t``; naming a series gives that observation's ascertainment predictor.

    Switching ``fixed`` / ``random`` / ``autocor`` off drops the covariate
    effects, the group-specific effects and the random walk respectively -- how
    you see, say, an ascertainment rate with its day-of-week effects removed.

    Parameters
    ----------
    idata : arviz.InferenceData
    panel : epidemia.core.PanelData
    config : epidemia.core.EpiModelConfig
    series : str | None
        Observation series name, or ``None`` for the ``R_t`` predictor.
    transform : bool
        Apply the inverse link. For a ``scaled_logit`` link that means
        ``K * logit^-1(eta)`` -- the cap IS applied, so an infection fatality
        ratio comes back on its natural scale (~0.01, not ~0.5) and must not be
        multiplied by ``link_K`` again. R's ``transform_rt`` does the same
        (``R/posterior_linpred.R:113-115``).

    Returns
    -------
    numpy.ndarray
        ``(draws, regions, time)``.
    """
    from .forecast import _apply_link

    post = idata.posterior
    M, T, K = np.shape(panel.X)

    if series is None:
        eta = np.zeros((_flat(post["Rt_unadj"]).shape[0], M, T))
        # R decomposes the predictor into pp_eta_fe() and pp_eta_re() and selects
        # them independently (R/pp_eta.R:26-63). The intercept belongs to the
        # FIXED part -- it is a column of object$fe -- and the region deviations
        # b belong to the RANDOM part alongside b0. Adding the intercept
        # unconditionally, or folding b in under `fixed`, makes these switches
        # decompose something other than the model.
        X = np.asarray(panel.X)
        # build_epidemia_model subtracts the design's column means when
        # center=True, so the coefficients are on the centred scale. Rebuilding
        # from the RAW design offsets the whole predictor by mean(X) . beta.
        if getattr(config, "center", False) and K:
            X = X - X.reshape(-1, K).mean(axis=0)[None, None, :]
        if fixed:
            if "intercept" in post:
                eta += _flat(post["intercept"])[:, None, None]
            if K and "beta" in post:
                beta = _flat(post["beta"])                  # (S, K)
                eta += np.einsum("mtk,sk->smt", X, beta)
        if random:
            if "b0" in post:
                eta += _flat(post["b0"])[:, :, None]
            if K and "b" in post:
                eta += np.einsum("mtk,smk->smt", X, _flat(post["b"]))
        if autocor:
            idxs = _rw_indices(getattr(config, "rw", None), panel)
            if idxs:
                eta += _walk_contribution(post, idxs, M, prefix="")
        link, cap = getattr(config, "link", "scaled_logit"), config.R_link_K
    else:
        from .core import ObsModel

        if obs_models is None:
            raise ValueError(
                "reconstructing an observation predictor needs the ObsModel it "
                "came from; pass obs_models="
            )
        if isinstance(obs_models, ObsModel):
            obs_models = [obs_models]
        try:
            o = next(m for m in obs_models if m.name == series)
        except StopIteration:
            raise ValueError(
                f"no series named {series!r}; have "
                f"{[m.name for m in obs_models]}"
            ) from None

        n_draws = _flat(post[f"{series}|rate"]).shape[0]
        eta = np.zeros((n_draws, M, T))
        if fixed and o.intercept and f"{series}|intercept" in post:
            eta += _flat(post[f"{series}|intercept"])[:, None, None]
        if fixed and o.X is not None and f"{series}|coef" in post:
            oX = np.asarray(o.X, dtype=float)
            if getattr(o, "center", False):
                oX = oX - oX.reshape(-1, oX.shape[2]).mean(axis=0)[None, None, :]
            coef = _flat(post[f"{series}|coef"])            # (S, Ks)
            eta += np.einsum("mtk,sk->smt", oX, coef)
        if o.offset is not None:
            eta += np.asarray(o.offset, dtype=float)[None, :, :]
        if autocor and o.rw is not None:
            idxs = _rw_indices(o.rw, panel)
            if idxs:
                eta += _walk_contribution(post, idxs, M, prefix=f"{series}|")
        link, cap = getattr(o, "link", "scaled_logit"), o.link_K

    return _apply_link(link, eta, cap) if transform else eta


def _rw_indices(rw, panel):
    """The per-term day->step index for one or more walks.

    The index lives on the :class:`~epidemia.core.RandomWalk` itself, which is
    where the model builder reads it (``core._random_walk``). Reading
    ``panel.rw_index`` instead -- an attribute only ``prepare_panel(rw_by=)``
    ever attaches -- silently dropped the walk for every model whose panel was
    built by hand, which is what both shipped tutorials do.
    """
    if rw is not None:
        terms = rw if isinstance(rw, (list, tuple)) else [rw]
        idx = [np.asarray(t.index, dtype=int) for t in terms
               if getattr(t, "index", None) is not None]
        if idx:
            return idx
    fallback = getattr(panel, "rw_index", None)
    return [np.asarray(fallback, dtype=int)] if fallback is not None else []


def _walk_contribution(post, indices, M, prefix=""):
    """Sum every random walk on one predictor, R's ``pp_eta_ac`` for both sides.

    ``core._walks`` names the first walk ``<prefix>rw`` and later ones
    ``<prefix>rw2``, ``rw3``, ... Each term carries its OWN day->step index, so
    they are zipped rather than sharing one.
    """
    total = 0.0
    for i, idx in enumerate(indices, start=1):
        name = f"{prefix}rw" if i == 1 else f"{prefix}rw{i}"
        if name not in post:
            break
        walk = _flat(post[name])                            # (S, procs, steps)
        # -1 means "no walk term on this day"; core pads a zero level for it.
        pad = np.concatenate(
            [np.zeros((walk.shape[0], walk.shape[1], 1)), walk], axis=2)
        proc = np.arange(M) if walk.shape[1] == M else np.zeros(M, int)
        total = total + pad[:, proc[:, None], np.asarray(idx) + 1]
    return total


def posterior_infectious(idata, config, normalise=True):
    """Total infectiousness -- R's ``posterior_infectious``.

    R computes ``(sum_s i_{t-s} gen_s) / max(gen)`` (``epidemia_pp_base.stan``
    divides the convolution by the modal mass of the generation kernel), so the
    series is on the scale of "infections weighted by how infectious they still
    are" rather than raw convolution units. Since ``gen`` is a simplex, ``max(gen)``
    is well under 1 -- typically ~0.14 for a serial interval -- so omitting the
    division inflates every value by ~7x. Returns ``(draws, regions, time)``.

    Parameters
    ----------
    normalise : bool, default True
        Divide by ``max(gen)``, matching R. Pass ``False`` for the raw
        convolution ``sum_s i_{t-s} gen_s``.
    """
    inf = _flat(idata.posterior["infections"])
    gen = np.asarray(config.gen, dtype=float)
    T = inf.shape[-1]
    out = np.zeros_like(inf)
    # lag-1-first, as everywhere else: gen[0] weights yesterday
    for k in range(1, min(len(gen) + 1, T)):
        out[..., k:] += gen[k - 1] * inf[..., : T - k]
    if normalise:
        gmax = float(np.max(gen))
        if gmax > 0:
            out = out / gmax
    return out


#: Which variables belong to each of R's ``par_types``. R's as.matrix.epimodel
#: restricts par_types to these names (R/plots.R:151-229); the patterns below map
#: them onto the variable names build_epidemia_model records.
PAR_TYPES = {
    "fixed": ("intercept", "beta", "coef"),
    "random": ("b0", "b", "Sigma", "sd", "z"),
    "autocor": ("rw", "rw_scale", "rw_noise"),
    "aux": ("aux", "inf_aux"),
    "seeds": ("seed", "seed_raw", "seed_tau", "seeds_aux"),
    "latent": ("infections_raw",),
}


def _matches_type(name, kind):
    """True when a variable name belongs to ``kind``."""
    base = str(name).split("|")[-1]
    pats = PAR_TYPES[kind]
    if kind == "autocor":
        return base == "rw" or base.startswith("rw")
    if kind == "aux":
        return base.endswith("aux") or base.endswith("aux_raw")
    if kind == "fixed":
        return base in ("intercept", "beta", "coef")
    if kind == "seeds":
        return base.startswith("seed")
    if kind == "random":
        return base in ("b0", "b") or base.startswith(("Sigma", "sd_", "z_"))
    return base in pats


def extract_samples(idata, pars=None, regex=None, series=None, groups=None,
                    par_types=None):
    """Draws as a flat ``(draws, parameters)`` frame -- R's ``as.matrix.epimodel``.

    Select by explicit names (``pars``), a regular expression (``regex``), an
    observation series, a list of regions, or by ``par_types`` -- R's
    ``c("fixed", "random", "autocor", "aux", "seeds", "latent")``, the keys of
    :data:`PAR_TYPES`.
    """
    import pandas as pd

    post = idata.posterior
    names = list(post.data_vars)
    if pars is not None:
        keep = [n for n in names if n in set(pars)]
    elif regex is not None:
        import re

        rx = re.compile(regex)
        keep = [n for n in names if rx.search(str(n))]
    elif series is not None:
        keep = [n for n in names if str(n).startswith((f"{series}|", f"E_{series}"))]
    else:
        keep = names

    if par_types is not None:
        kinds = [par_types] if isinstance(par_types, str) else list(par_types)
        unknown = set(kinds) - set(PAR_TYPES)
        if unknown:
            raise ValueError(
                f"unknown par_types {sorted(unknown)}; choose from "
                f"{sorted(PAR_TYPES)}"
            )
        keep = [n for n in keep if any(_matches_type(n, k) for k in kinds)]

    cols = {}
    for n in keep:
        a = _flat(post[n])
        if a.ndim == 1:
            cols[str(n)] = a
            continue
        flat = a.reshape(a.shape[0], -1)
        labels = _labels(post[n], groups)
        for j, lab in enumerate(labels):
            cols[f"{n}[{lab}]"] = flat[:, j]
    frame = pd.DataFrame(cols)
    if groups is not None:
        wanted = set(groups)
        frame = frame[[c for c in frame.columns
                       if "[" not in c or any(g in c for g in wanted)]]
    return frame


def _labels(da, groups=None):
    """Column labels for a multi-dimensional posterior variable."""
    dims = [d for d in da.dims if d not in ("chain", "draw")]
    coords = [list(map(str, da.coords[d].values)) if d in da.coords
              else list(map(str, range(da.sizes[d]))) for d in dims]
    if not coords:
        return [""]
    out = coords[0]
    for c in coords[1:]:
        out = [f"{a},{b}" for a in out for b in c]
    return out


class PriorSummary:
    """What :func:`prior_summary` returns; print it."""

    def __init__(self, rows):
        self._rows = rows

    def __repr__(self):
        width = max((len(r[1]) for r in self._rows), default=0)
        lines = []
        current = None
        for regression, name, spec in self._rows:
            if regression != current:
                lines.append(f"\n{regression}")
                lines.append("-" * len(regression))
                current = regression
            lines.append(f"  {name:<{width}}  {spec}")
        return "\n".join(lines).lstrip("\n")

    __str__ = __repr__

    def as_rows(self):
        """The underlying ``(regression, parameter, description)`` triples."""
        return list(self._rows)


def prior_summary(panel, obs_models, config):
    """The priors actually in force, per regression -- R's ``prior_summary``.

    Reads the configuration rather than the fit, so it works before sampling.
    Fields left as ``None`` report the scalar-hyperparameter default that will
    be used instead.
    """
    from .core import ObsModel

    def describe(spec, fallback):
        if spec is None:
            return fallback
        params = getattr(spec, "params", lambda: {})()
        inner = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{getattr(spec, 'dist', type(spec).__name__)}({inner})"

    rows = []
    K = np.shape(panel.X)[2]
    if config.intercept:
        rows.append(("R_t", "intercept",
                     describe(config.prior_intercept, "normal(0, 0.5)")))
    if K:
        rows.append(("R_t", "covariates", describe(
            config.prior_covariates,
            f"shifted_gamma(shape={config.beta_shape:.4g}, "
            f"scale={config.beta_scale:.4g}, shift={config.beta_shift:.4g})")))
    if config.region_effects:
        rows.append(("R_t", "region effects", describe(
            config.prior_covariance,
            "correlated: LKJ + Gamma scales" if config.correlated
            else f"independent Gamma(shape=[{config.sd_intercept_shape:.4g}, "
                 f"{config.sd_slope_shape:.4g}...], scale={config.sd_scale:.4g})")))
    if config.rw is not None:
        # EpiModelConfig.rw may be a LIST of walks (core._walks sums them);
        # reading .prior_scale off the list raised AttributeError.
        _walk_terms = (config.rw if isinstance(config.rw, (list, tuple))
                       else [config.rw])
        scales = ", ".join(f"{t.prior_scale:.4g}" for t in _walk_terms)
        per_region = any(getattr(t, "by_region", False) for t in _walk_terms)
        rows.append(("R_t", "random walk",
                     f"HalfNormal({scales}) on the step size"
                     + (" (one per region)" if per_region else " (shared)")))

    # role_scale is what core.py builds with, so report the SAME prior rather
    # than the unscaled spec -- they differ 4x for a bare normal().
    from . import priors as _pr

    rows.append(("infections", "seeds", describe(
        _pr.role_scale(config.prior_seeds, "seeds"),
        f"hexp(exponential({config.seed_aux_rate:.4g}))" if config.seed_pooling
        else f"exponential(1/{config.seed_prior_mean:.4g})")))
    if config.latent:
        rows.append(("infections", "dispersion", describe(
            config.prior_aux,
            f"{config.latent_aux_loc:.4g} + {config.latent_aux_scale:.4g}"
            " * HalfNormal(1)")))

    if isinstance(obs_models, ObsModel):
        obs_models = [obs_models]
    for o in obs_models:
        if o.intercept:
            rows.append((o.name, "intercept", describe(
                o.prior_intercept, f"normal(0, {o.prior_intercept_scale:.4g})")))
        if o.X is not None:
            rows.append((o.name, "coefficients", describe(
                o.prior, f"normal(0, {o.prior_coef_scale:.4g})")))
        if o.family != "poisson":
            rows.append((o.name, "auxiliary", describe(
                o.prior_aux,
                f"{o.prior_aux_loc:.4g} + {o.prior_aux_scale:.4g} * HalfNormal(1)")))
    return PriorSummary(rows)


def summary(idata, pars=None, regex=None, series=None, groups=None,
            par_types=None, probs=(0.1, 0.5, 0.9), digits=2):
    """Parameter summary with diagnostics -- R's ``summary.epimodel``.

    Mean, SD, the requested quantiles, Monte-Carlo standard error, effective
    sample size and R-hat, for whichever parameters the selectors keep. The
    selection arguments are those of :func:`extract_samples`, so
    ``par_types="fixed"`` gives just the fixed effects, as in R.

    Parameters
    ----------
    idata : arviz.InferenceData
    pars, regex, series, groups, par_types
        Parameter selection; see :func:`extract_samples`.
    probs : sequence[float]
        Quantiles to report. R's default is ``(0.1, 0.5, 0.9)``.
    digits : int
        Round the result. ``None`` leaves it unrounded.

    Returns
    -------
    pandas.DataFrame
        One row per parameter, indexed by name.
    """
    import warnings

    import arviz as az
    import pandas as pd

    draws = extract_samples(idata, pars=pars, regex=regex, series=series,
                            groups=groups, par_types=par_types)
    if not len(draws.columns):
        raise ValueError("no parameters matched the selection")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = az.summary(idata, kind="diagnostics")

    rows = {}
    for col in draws.columns:
        v = draws[col].to_numpy(dtype=float)
        row = {"mean": v.mean(), "sd": v.std(ddof=1)}
        for q in probs:
            row[f"{100 * q:g}%"] = np.quantile(v, q)
        for stat in ("mcse_mean", "ess_bulk", "ess_tail", "r_hat"):
            row[stat] = (float(diag.loc[col, stat])
                         if col in diag.index and stat in diag.columns
                         else np.nan)
        rows[col] = row

    out = pd.DataFrame.from_dict(rows, orient="index")
    return out if digits is None else out.round(digits)
