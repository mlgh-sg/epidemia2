"""Multilevel (partially-pooled) renewal model with covariates.

This extends the single-population core in :mod:`epidemia.model` to the setting
of the R package's *multilevel* / *partial-pooling* example: several regions are
fit jointly, region-specific reproduction numbers depend on covariates (here the
non-pharmaceutical interventions, NPIs), and the covariate effects are
**partially pooled** across regions.

For region :math:`m` and day :math:`t` the reproduction number is

.. math::

    R^{(m)}_t = K\\,\\operatorname{sigmoid}\\!\\Big(b^{(m)}_0
        + \\sum_k \\big(\\beta_k + b^{(m)}_k\\big) X^{(m)}_{k,t}\\Big),

where :math:`\\beta_k` are **fixed** (global) NPI effects and :math:`b^{(m)}_k`
are **partially-pooled** region effects with :math:`b^{(m)}_k \\sim N(0,
\\sigma_k)`.  Latent infections follow the same renewal recursion as the
single-population model, run in parallel across regions, and expected deaths are
an infection-to-death convolution scaled by a (pooled) infection-fatality ratio.

The model is declared in PyMC and fit with **nutpie** (NUTS / MCMC) via
:func:`fit_multilevel` -- unlike the R vignette, which used Variational Bayes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MultilevelConfig:
    """Configuration for the partially-pooled multi-region renewal model.

    The defaults reproduce the priors and links used in the R package's Europe /
    COVID-19 multilevel example.

    Attributes
    ----------
    gen : array (L,)
        Generation-interval kernel; ``gen[s]`` weights infections ``s+1`` days ago.
    i2o : array (K,)
        Infection-to-death delay PMF; ``i2o[k]`` weights infections ``k`` days ago.
    R_link_K : float
        Carrying capacity of the scaled-logit link on ``R`` (``R = K*sigmoid(eta)``).
    ifr_link_K : float
        Upper bound of the infection-fatality ratio (``IFR = K*sigmoid(alpha)``).
    beta_shape, beta_scale, beta_shift : float
        Shifted-gamma prior on the fixed NPI effects: ``beta_k = beta_shift - g_k``
        with ``g_k ~ Gamma(beta_shape, scale=beta_scale)`` (so effects are a priori
        non-positive up to a small ``beta_shift``).
    sd_intercept_shape, sd_slope_shape, sd_scale : float
        Gamma priors on the between-region standard deviations: the intercept SD
        is ``Gamma(sd_intercept_shape, scale=sd_scale)`` and each slope SD is
        ``Gamma(sd_slope_shape, scale=sd_scale)``.
    ifr_intercept_scale : float
        SD of the ``Normal(0, .)`` prior on the IFR intercept ``alpha``.
    seed_days : int
        Number of initial days over which infections are seeded (per region).
    seed_prior_mean : float
        Mean of the exponential prior on each region's seeded infections.
    """

    gen: np.ndarray
    i2o: np.ndarray
    R_link_K: float = 6.5
    ifr_link_K: float = 0.02
    beta_shape: float = 1.0 / 6.0
    beta_scale: float = 1.0
    beta_shift: float = float(np.log(1.05) / 6.0)
    sd_intercept_shape: float = 2.0
    sd_slope_shape: float = 0.5
    sd_scale: float = 0.25
    ifr_intercept_scale: float = 0.2
    seed_days: int = 6
    seed_prior_mean: float = 30.0
    _extra: dict = field(default_factory=dict, repr=False)


@dataclass
class MultilevelData:
    """Padded, model-ready arrays for the multi-region renewal model.

    All region series are left-aligned to their own first modelled day and padded
    on the right to a common length ``T``; ``mask`` marks the genuine days.

    Attributes
    ----------
    deaths : int array (M, T)
        Daily deaths per region (0 where padded).
    X : float array (M, T, K)
        NPI design matrix per region and day.
    mask : bool array (M, T)
        ``True`` on genuine (non-padded) days.
    lengths : int array (M,)
        Number of genuine days per region.
    regions : list[str]
        Region names, in row order.
    npis : list[str]
        Covariate (NPI) names, in column order of ``X``.
    dates : list[numpy.ndarray]
        The genuine dates for each region (length ``lengths[m]``).
    """

    deaths: np.ndarray
    X: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    regions: list
    npis: list
    dates: list


def prepare_panel(df, npis, response="deaths", group="country", date="date",
                  seed_offset=30, death_threshold=10, fit_until=None):
    """Turn the long ``EuropeCovid2`` panel into padded model arrays.

    Mirrors the R vignette's data preparation: for each region the modelled
    window begins ``seed_offset`` days before cumulative deaths first exceed
    ``death_threshold`` and (optionally) ends strictly before ``fit_until``.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel (e.g. ``epidemia.europe_covid2().data``).
    npis : list[str]
        Covariate columns to place in the design matrix.
    response, group, date : str
        Column names for the observed counts, the region factor and the date.
    seed_offset : int
        Days before the ``death_threshold`` crossing at which to begin seeding.
    death_threshold : int
        Cumulative-death threshold used to choose each region's start date.
    fit_until : str | None
        If given (``"YYYY-MM-DD"``), keep only days strictly before this date.

    Returns
    -------
    MultilevelData
    """
    import pandas as pd

    cutoff = pd.Timestamp(fit_until) if fit_until is not None else None
    per_region = {}
    for name, g in df.sort_values(date).groupby(group, sort=True):
        g = g.reset_index(drop=True)
        crossings = np.where(g[response].cumsum().to_numpy() > death_threshold)[0]
        if len(crossings) == 0:
            continue
        start_row = crossings[0] - seed_offset
        start_date = g[date].iloc[start_row]
        g = g[g[date] > start_date]
        if cutoff is not None:
            g = g[g[date] < cutoff]
        if len(g) > 0:
            per_region[str(name)] = g.reset_index(drop=True)

    regions = list(per_region)
    lengths = np.array([len(per_region[r]) for r in regions])
    M, T, K = len(regions), int(lengths.max()), len(npis)

    deaths = np.zeros((M, T), dtype=int)
    X = np.zeros((M, T, K), dtype=float)
    mask = np.zeros((M, T), dtype=bool)
    dates = []
    for m, r in enumerate(regions):
        g = per_region[r]
        n = len(g)
        deaths[m, :n] = np.nan_to_num(g[response].to_numpy(dtype=float)).astype(int)
        X[m, :n, :] = g[npis].to_numpy(dtype=float)
        mask[m, :n] = True
        dates.append(g[date].to_numpy())

    return MultilevelData(deaths=deaths, X=X, mask=mask, lengths=lengths,
                          regions=regions, npis=list(npis), dates=dates)


def build_multilevel_model(data: MultilevelData, config: MultilevelConfig):
    """Build the partially-pooled multi-region renewal model (a ``pymc.Model``)."""
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    gen = np.asarray(config.gen, dtype=float)
    i2o = np.asarray(config.i2o, dtype=float)
    L = gen.shape[0]
    v = int(config.seed_days)

    deaths = np.asarray(data.deaths)
    X = np.asarray(data.X, dtype=float)
    mask = np.asarray(data.mask)
    M, T, K = X.shape

    coords = {"region": data.regions, "npi": data.npis, "region_time": np.arange(T)}
    with pm.Model(coords=coords) as model:
        # --- Transmission: fixed NPI effects (shifted gamma) + pooled region effects
        g_beta = pm.Gamma("g_beta", alpha=config.beta_shape,
                          beta=1.0 / config.beta_scale, dims="npi")
        beta = pm.Deterministic("beta", config.beta_shift - g_beta, dims="npi")

        sd_shapes = np.concatenate([[config.sd_intercept_shape],
                                    np.full(K, config.sd_slope_shape)])
        sd = pm.Gamma("sd", alpha=sd_shapes, beta=1.0 / config.sd_scale,
                      shape=K + 1)  # [intercept, slope_1, ..., slope_K]

        z0 = pm.Normal("z0", 0.0, 1.0, dims="region")            # non-centred
        z = pm.Normal("z", 0.0, 1.0, dims=("region", "npi"))
        b0 = pm.Deterministic("b0", sd[0] * z0, dims="region")   # region intercepts
        b = pm.Deterministic("b", sd[1:] * z, dims=("region", "npi"))  # region slopes

        # eta[m, t] = b0[m] + sum_k (beta_k + b[m, k]) * X[m, t, k]
        Xt = pt.as_tensor_variable(X)                            # (M, T, K)
        coef = beta[None, :] + b                                 # (M, K)
        eta = b0[:, None] + (Xt * coef[:, None, :]).sum(axis=2)  # (M, T)
        R = pm.Deterministic("Rt", config.R_link_K * pt.sigmoid(eta),
                             dims=("region", "region_time"))

        # --- Infections: renewal recursion, run in parallel across regions
        seed = pm.Exponential("seed", 1.0 / config.seed_prior_mean, dims="region")
        seeds = pt.outer(seed, pt.ones(v))                       # (M, v), constant
        buf0 = pt.zeros((M, L))
        buf0 = pt.set_subtensor(buf0[:, :min(v, L)], seed[:, None])

        def step(R_t, buf):                                      # R_t: (M,), buf: (M, L)
            i_t = R_t * pt.dot(buf, gen)                         # (M,)
            new_buf = pt.concatenate([i_t[:, None], buf[:, :-1]], axis=1)
            return new_buf, i_t

        _, infs = pytensor.scan(
            fn=step, sequences=[R[:, v:].T], outputs_info=[buf0, None],
            return_updates=False,
        )                                                        # infs: (T - v, M)
        infections = pt.concatenate([seeds, infs.T], axis=1)     # (M, T)
        pm.Deterministic("infections", infections, dims=("region", "region_time"))

        # --- Expected deaths: IFR * (infection-to-death convolution), IFR pooled
        alpha = pm.Normal("alpha", 0.0, config.ifr_intercept_scale)
        ifr = pm.Deterministic("ifr", config.ifr_link_K * pt.sigmoid(alpha))
        # Causal convolution; taps beyond the modelled window reference
        # pre-window infections we don't model, so cap the kernel at T.
        terms = [i2o[0] * infections]
        for k in range(1, min(len(i2o), T)):
            terms.append(i2o[k] * pt.concatenate(
                [pt.zeros((M, k)), infections[:, :T - k]], axis=1))
        conv = pt.add(*terms) if len(terms) > 1 else terms[0]
        E = pm.Deterministic("E_deaths", ifr * conv + 1e-6,
                             dims=("region", "region_time"))

        # --- Likelihood on genuine days only (negative binomial, as in R)
        idx = np.where(mask.reshape(-1))[0]
        mu = E.reshape((-1,))[idx]
        phi = pm.HalfNormal("reciprocal_dispersion", 5.0)
        pm.NegativeBinomial("deaths", mu=mu, alpha=phi,
                            observed=deaths.reshape(-1)[idx])

    return model


def fit_multilevel(data: MultilevelData, config: MultilevelConfig,
                   draws=1000, tune=1000, chains=4, seed=0,
                   adaptation="diag", backend="numba", progress_bar=False, **kwargs):
    """Fit the multi-region renewal model with nutpie (NUTS / MCMC).

    Parameters mirror :func:`epidemia.fit`. Returns an ArviZ ``InferenceData``
    whose posterior holds the fixed effects ``beta``, region effects ``b0``/``b``,
    between-region SDs ``sd``, ``ifr`` and the latent ``Rt``/``infections``/
    ``E_deaths`` series (indexed by ``region``).
    """
    import nutpie

    model = build_multilevel_model(data, config)
    kw = {}
    if backend == "jax":
        kw["gradient_backend"] = "jax"
    compiled = nutpie.compile_pymc_model(model, backend=backend, **kw)
    idata = nutpie.sample(
        compiled, draws=draws, tune=tune, chains=chains, seed=seed,
        adaptation=adaptation, progress_bar=progress_bar, **kwargs,
    )
    return idata
