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

import warnings
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
        Infection-to-death delay PMF; ``i2o[k]`` weights infections ``k+1`` days
        ago. Both kernels are lag-1-first, as in R's Stan code, so vectors taken
        straight from the R data objects (``EuropeCovid2$si``,
        ``EuropeCovid2$inf2death``) drop in unchanged.
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
        ``Gamma(sd_slope_shape, scale=sd_scale)``. Estimating these SDs is what
        makes the region effects *partially pooled*.
    sd_slope_fixed : float | sequence | None
        If given, **do not estimate** the between-region SDs of the *covariate
        slopes* -- hold them fixed at this value (a scalar broadcasts to all ``K``
        slopes; a sequence gives one per covariate). This is how you select the
        no-pooling and full-pooling regimes; see :meth:`pooling`. ``None``
        (default) estimates them, i.e. partial pooling. The **region-intercept**
        SD is always estimated: as in R, how the intercepts are specified is a
        separate axis from how the covariate effects are pooled, and each region
        needs its own baseline ``R_0``.
    ifr_intercept_scale : float
        SD of the ``Normal(0, .)`` prior on the IFR intercept ``alpha``.
    seed_days : int
        Number of initial days over which infections are seeded (per region).
    seed_pooling : bool
        If ``True`` (default, and R's ``epiinf`` default) the seeded infections
        are **partially pooled** across regions through a shared mean ``tau``:
        ``tau ~ Exponential(seed_aux_rate)`` and ``seed[m] | tau ~
        Exponential(mean=tau)``. This is R's ``prior_seeds = hexp(prior_aux =
        exponential(0.03))``. Regions with little early death data then borrow
        epidemic-size information from the others instead of leaning entirely on
        a fixed prior mean. If ``False``, each region gets an independent
        ``Exponential(mean=seed_prior_mean)``.
    seed_aux_rate : float
        Rate of the ``Exponential`` prior on the shared seed mean ``tau`` when
        ``seed_pooling`` is on; ``0.03`` (R's default) gives ``tau`` a prior mean
        of ~33 infections/day.
    seed_prior_mean : float
        Mean of each region's exponential seed prior when ``seed_pooling`` is off.
    dispersion_loc, dispersion_scale : float
        The negative-binomial ``reciprocal_dispersion`` is
        ``dispersion_loc + dispersion_scale * HalfNormal(1)``, i.e.
        ``10 + 5 * HalfNormal(1)`` -- R's ``epiobs`` default
        ``prior_aux = normal(location = 10, scale = 5)`` truncated to be positive.
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
    sd_slope_fixed: object = None
    ifr_intercept_scale: float = 0.2
    seed_days: int = 6
    seed_pooling: bool = True
    seed_aux_rate: float = 0.03
    seed_prior_mean: float = 30.0
    dispersion_loc: float = 10.0
    dispersion_scale: float = 5.0
    _extra: dict = field(default_factory=dict, repr=False)

    def pooling(self, regime: str, slope_sd: float = 5.0):
        """A copy of this config in one of the three pooling regimes.

        Only the pooling of the **covariate effects** changes; the region
        intercepts stay partially pooled with their SD estimated in every regime,
        so each region keeps its own baseline ``R_0``. That mirrors R, where the
        intercept specification and the covariate pooling are independent parts of
        the formula.

        Parameters
        ----------
        regime : {"partial", "none", "full"}
            * ``"partial"`` -- estimate the between-region slope SDs from the data
              (the default, and what the R ``(npi || region)`` term does): each
              region's effect is shrunk toward the global mean by an amount the
              data decide.
            * ``"none"`` -- fix the slope SDs at ``slope_sd``, large enough that
              the prior barely constrains ``b[m, k]``, so every region's effect is
              effectively free and no information is shared.
            * ``"full"`` -- fix the slope SDs at ~0, collapsing every region's
              effect onto the single global ``beta_k``.
        slope_sd : float
            The fixed slope SD used by ``"none"``. The default of 5 is wide on the
            logit-``R_t`` scale (the whole link spans roughly +/-4) without being
            wide enough to saturate it.

        Notes
        -----
        Prefer this over hand-setting a huge ``sd_slope_shape``. Inflating the
        *Gamma prior's shape* to fake no-pooling (e.g. ``sd_slope_shape=1e6``)
        puts the between-region SD near 1e6, so ``b = sd * z`` lands around
        ``N(0, 1e6)``, ``eta`` saturates the ``scaled_logit`` link, and every
        ``R_t`` collapses to exactly 0 or ``R_link_K``. That is a broken prior,
        not an unpooled model: the fit does not converge and the resulting effect
        estimates are ~1e6 in magnitude, which silently destroys the axis of any
        plot they share with a well-behaved fit.
        """
        from dataclasses import replace

        if regime == "partial":
            return replace(self, sd_slope_fixed=None)
        if regime == "none":
            return replace(self, sd_slope_fixed=float(slope_sd))
        if regime == "full":
            return replace(self, sd_slope_fixed=1e-6)
        raise ValueError(f"regime must be 'partial', 'none' or 'full'; got {regime!r}")


@dataclass
class MultilevelData:
    """Padded, model-ready arrays for the multi-region renewal model.

    All region series are left-aligned to their own first modelled day and padded
    on the right to a common length ``T``; ``mask`` marks the genuine days.

    Attributes
    ----------
    deaths : int array (M, T)
        Daily deaths per region (0 where padded or missing -- see ``mask``).
    X : float array (M, T, K)
        NPI design matrix per region and day.
    mask : bool array (M, T)
        ``True`` on days that enter the **likelihood**: genuine (non-padded) days
        whose response was actually observed. Days inside a region's window whose
        count was missing are ``False`` here -- their latent ``Rt``/``infections``
        are still modelled, they simply contribute no likelihood term. So
        ``mask.sum(axis=1) <= lengths``, with equality iff nothing is missing.
    lengths : int array (M,)
        Number of genuine (modelled) days per region -- the length of each
        region's series, missing observations included. Use this, not ``mask``,
        to slice a region's latent series for plotting.
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
        if start_row < 0:
            # Not enough lead-in before the threshold crossing. A negative
            # .iloc would silently wrap to the END of the series and keep only
            # the last few days, so clamp to the first available day instead and
            # say so -- the region simply gets a shorter seeding window.
            warnings.warn(
                f"{name!r}: only {crossings[0]} day(s) of data before cumulative "
                f"{response} exceeded {death_threshold}, fewer than seed_offset="
                f"{seed_offset}; starting at the first available day instead.",
                stacklevel=2,
            )
            start_row = 0
            g = g[g[date] >= g[date].iloc[0]]
        else:
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
    n_missing = 0
    for m, r in enumerate(regions):
        g = per_region[r]
        n = len(g)
        y = g[response].to_numpy(dtype=float)
        observed = np.isfinite(y)  # a missing count is UNOBSERVED, not a zero
        n_missing += int((~observed).sum())
        deaths[m, :n] = np.nan_to_num(y).astype(int)
        X[m, :n, :] = g[npis].to_numpy(dtype=float)
        mask[m, :n] = observed
        dates.append(g[date].to_numpy())
    if n_missing:
        warnings.warn(
            f"{n_missing} missing {response} value(s) inside the modelled window "
            "were left out of the likelihood (masked as unobserved, as R treats "
            "NA). The latent series are still estimated on those days.",
            stacklevel=2,
        )

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

        # Between-region SDs: sd = [intercept, slope_1, ..., slope_K].
        # Slope SDs estimated => partial pooling; fixed => no/full pooling (see
        # MultilevelConfig.pooling). The intercept SD is estimated either way --
        # fixing it too would delete each region's own baseline R_0, which is a
        # different modelling choice from how the covariate effects are pooled.
        if config.sd_slope_fixed is None:
            sd_shapes = np.concatenate([[config.sd_intercept_shape],
                                        np.full(K, config.sd_slope_shape)])
            sd = pm.Gamma("sd", alpha=sd_shapes, beta=1.0 / config.sd_scale,
                          shape=K + 1)
        else:
            sd0 = pm.Gamma("sd_intercept", alpha=config.sd_intercept_shape,
                           beta=1.0 / config.sd_scale)
            slopes = np.broadcast_to(
                np.asarray(config.sd_slope_fixed, dtype=float), (K,)
            ).copy()
            sd = pm.Deterministic(
                "sd", pt.concatenate([sd0.reshape((1,)), pt.as_tensor_variable(slopes)])
            )

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
        if config.seed_pooling:
            # R's hexp: tau ~ Exponential(rate), seed[m] | tau ~ Exponential(mean=tau).
            # Written non-centred (tau * unit exponential), as R's Stan does.
            tau = pm.Exponential("seed_tau", config.seed_aux_rate)
            seed_raw = pm.Exponential("seed_raw", 1.0, dims="region")
            seed = pm.Deterministic("seed", tau * seed_raw, dims="region")
        else:
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
        # Causal convolution at lags 1..K: i2o[k] weights infections k+1 days
        # earlier, matching `gen` above and R's Stan (which sums over
        # infections[start .. t-1] and never lag 0). Taps beyond the modelled
        # window would reference pre-window infections we don't model, so cap
        # the kernel at T.
        terms = []
        for k in range(1, min(len(i2o) + 1, T)):
            terms.append(i2o[k - 1] * pt.concatenate(
                [pt.zeros((M, k)), infections[:, :T - k]], axis=1))
        conv = pt.add(*terms) if len(terms) > 1 else terms[0]
        E = pm.Deterministic("E_deaths", ifr * conv + 1e-6,
                             dims=("region", "region_time"))

        # --- Likelihood on genuine days only (negative binomial, as in R)
        idx = np.where(mask.reshape(-1))[0]
        mu = E.reshape((-1,))[idx]
        # R's epiobs default: prior_aux = normal(10, 5) on a lower=0 parameter,
        # i.e. reciprocal_dispersion = 10 + 5 * HalfNormal(1).
        phi_raw = pm.HalfNormal("reciprocal_dispersion_raw", 1.0)
        phi = pm.Deterministic(
            "reciprocal_dispersion",
            config.dispersion_loc + config.dispersion_scale * phi_raw,
        )
        pm.NegativeBinomial("deaths", mu=mu, alpha=phi,
                            observed=deaths.reshape(-1)[idx])

    return model


def _coef_draws(idata):
    """``(draws, M, K)`` region-specific coefficients ``beta_k + b[m, k]``."""
    post = idata.posterior
    beta = np.moveaxis(np.asarray(post["beta"].stack(s=("chain", "draw"))), -1, 0)
    b = np.moveaxis(np.asarray(post["b"].stack(s=("chain", "draw"))), -1, 0)
    b0 = np.moveaxis(np.asarray(post["b0"].stack(s=("chain", "draw"))), -1, 0)
    return beta[:, None, :] + b, b0  # (D, M, K), (D, M)


def effect_table(idata, config, data=None, group=None, levels=90):
    """Percent reduction in ``R_t`` attributable to each measure, per region.

    The model's link is ``R = K * sigmoid(eta)``, **not** a log link, so a
    coefficient does *not* translate into a constant multiplicative effect and
    ``1 - exp(beta_k)`` is the wrong answer. This computes the reduction the way
    it is actually defined -- by counterfactual, per posterior draw:

    * ``R_0``          -- ``K * sigmoid(b0[m])``, transmission with no measures;
    * *measure k alone* -- ``K * sigmoid(b0[m] + beta_k + b[m,k])``, and the
      reduction is ``1 - R_k / R_0``;
    * *all measures*   -- every covariate switched on at once.

    Because the link is nonlinear, the same coefficient buys a different
    percentage in a region with a high ``R_0`` than in one with a low ``R_0``,
    which is why this is reported per region rather than as one global number.

    Parameters
    ----------
    idata : arviz.InferenceData
        A fit from :func:`fit_multilevel`.
    config : MultilevelConfig
        The config used for the fit (supplies ``R_link_K``).
    data : MultilevelData | None
        The panel that was fitted. Strongly recommended: it lets each row be
        flagged with whether that region ever actually enacted that measure.
        Where it did not (Sweden and lockdown, say), the percentage is a pure
        **counterfactual extrapolation** from the pooled prior -- the region's
        own data contain no information about it -- and ``enacted`` is ``False``.
    group : str | None
        Restrict to one region. ``None`` (default) does every region.
    levels : float
        Central credible-interval width, in percent.

    Returns
    -------
    pandas.DataFrame
        Columns ``region, term, kind, enacted, median, lo, hi``. ``kind="pct"``
        rows are percent reductions in ``R_t`` (positive = transmission reduced);
        ``kind="R"`` rows (``R_0``, ``R_t``) are reproduction numbers. Read
        ``enacted=False`` rows as "what this measure *would* have done here",
        not as a measured effect.
    """
    import pandas as pd

    K_link = config.R_link_K
    post = idata.posterior
    regions = [str(r) for r in post.coords["region"].values]
    npis = [str(n) for n in post.coords["npi"].values]
    keep = regions if group is None else [str(group)]
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    coef, b0 = _coef_draws(idata)  # (D, M, K), (D, M)
    lo_q, hi_q = (100 - levels) / 2, 100 - (100 - levels) / 2

    def sig(x):
        return K_link / (1.0 + np.exp(-x))

    def enacted_in(region, k):
        """Did this region ever actually switch measure k on?"""
        if data is None:
            return None
        m = data.regions.index(region)
        return bool(data.X[m, : int(data.lengths[m]), k].any())

    def row(region, term, d, kind="pct", enacted=None):
        return {"region": region, "term": term, "kind": kind, "enacted": enacted,
                "median": float(np.median(d)), "lo": float(np.percentile(d, lo_q)),
                "hi": float(np.percentile(d, hi_q))}

    rows = []
    for r in keep:
        m = regions.index(r)
        R_none = sig(b0[:, m])                       # (D,)
        rows.append(row(r, "R_0 (no measures)", R_none, kind="R"))
        for k, npi in enumerate(npis):
            R_k = sig(b0[:, m] + coef[:, m, k])
            rows.append(row(r, npi, 100.0 * (1.0 - R_k / R_none),
                            enacted=enacted_in(r, k)))
        R_all = sig(b0[:, m] + coef[:, m, :].sum(axis=1))
        # "all measures" is only a measured effect where the region used them all
        all_enacted = (None if data is None
                       else all(enacted_in(r, k) for k in range(len(npis))))
        rows.append(row(r, "all measures", 100.0 * (1.0 - R_all / R_none),
                        enacted=all_enacted))
        rows.append(row(r, "R_t (all measures)", R_all, kind="R",
                        enacted=all_enacted))
    return pd.DataFrame(rows)


def _compile(model, backend="numba", progress_bar=True, **sample_kw):
    """Compile ``model`` with nutpie and sample, announcing the compile step.

    Compilation of the log-density routinely takes as long as the sampling for
    these scan-based models and nutpie reports no progress during it, so an
    un-announced fit looks hung. We print around it and let nutpie's own bar
    cover the sampling.
    """
    import time

    import nutpie

    kw = {}
    if backend == "jax":
        kw["gradient_backend"] = "jax"
    if progress_bar:
        print(f"[epidemia] compiling the log-density ({backend} backend)...",
              end=" ", flush=True)
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(model, backend=backend, **kw)
    if progress_bar:
        print(f"done in {time.time() - t0:.1f}s", flush=True)
        chains, draws, tune = (sample_kw.get("chains"), sample_kw.get("draws"),
                               sample_kw.get("tune"))
        print(f"[epidemia] sampling {chains} chains x ({tune} tune + {draws} draws)",
              flush=True)
    t0 = time.time()
    idata = nutpie.sample(compiled, progress_bar=progress_bar, **sample_kw)
    if progress_bar:
        print(f"[epidemia] sampled in {time.time() - t0:.1f}s", flush=True)
    return idata


def fit_multilevel(data: MultilevelData, config: MultilevelConfig,
                   draws=1000, tune=1000, chains=4, seed=0, adaptation="low_rank",
                   backend="numba", progress_bar=True, target_accept=0.95, **kwargs):
    """Fit the multi-region renewal model with nutpie (NUTS / MCMC).

    Parameters mirror :func:`epidemia.fit`. Returns an ArviZ ``InferenceData``
    whose posterior holds the fixed effects ``beta``, region effects ``b0``/``b``,
    between-region SDs ``sd``, ``ifr`` and the latent ``Rt``/``infections``/
    ``E_deaths`` series (indexed by ``region``).

    ``progress_bar`` defaults to ``True``: this model takes minutes, so silence
    is unhelpful. The compile step is announced separately from the sampling.

    ``target_accept`` defaults to **0.95** rather than nutpie's 0.8. This is a
    hierarchical model whose funnel geometry (the between-region SDs against the
    non-centred region effects) makes 0.8 produce hundreds of divergences on the
    full 11-country example. A divergence check is run after sampling and warns
    if any remain.

    ``adaptation`` defaults to ``"low_rank"`` rather than nutpie's ``"diag"``,
    for the reason below.

    .. note:: **Why the mass-matrix default differs from nutpie's.**

       Divergences and ``r_hat`` are *different* failures with different cures,
       and this model can hit both. Collinear covariates (NPIs enacted within
       days of each other, say) give a posterior with a long thin correlation
       ridge that a diagonal mass matrix cannot follow: the sampler does not
       diverge, it simply fails to mix. On the 11-country example ``"diag"``
       shows up as ``r_hat`` 1.08-1.11 and an effective sample size of 25-36 out
       of 4000 for the collinear coefficients -- and their estimates are biased
       as a result. ``"low_rank"`` fits a low-rank correction and brings every
       ``r_hat`` to <= 1.04.

       Benchmarked, it is faster too, so this is not a speed/accuracy trade:
       against ``"diag"`` it gave 1.3x the wall-clock speed and 4.6x the
       effective samples per second on the 11-country model, and 4.7x / 6.3x on
       a small single-population one. No case was found where ``"diag"`` won.
       See ``benchmarks/`` and the docs' "Performance" page.

       Raising ``target_accept`` will not fix a ridge, and a clean divergence
       count is not evidence that one is absent. Always check ``arviz.summary``
       for ``r_hat`` and ``ess_bulk`` too.
    """
    model = build_multilevel_model(data, config)
    idata = _compile(model, backend=backend, draws=draws, tune=tune, chains=chains,
                     seed=seed, adaptation=adaptation, progress_bar=progress_bar,
                     target_accept=target_accept, **kwargs)
    _warn_on_divergences(idata)
    return idata


def _warn_on_divergences(idata):
    """Surface sampler problems at fit time -- silence here has cost people real results.

    Covers divergences, max-treedepth saturation, E-BFMI, R-hat and ESS, the
    same set R warns about in ``check_hmc_diagnostics()``. Call
    :func:`epidemia.sampler_diagnostics` on the returned fit for the detail.
    """
    from .diagnostics import sampler_diagnostics
    try:
        sampler_diagnostics(idata, warn=True)
    except (ValueError, KeyError, AttributeError):
        # No sample_stats: a variational fit, or something that is not a NUTS
        # run. Nothing to check rather than something to complain about.
        return
