"""The full renewal model: many regions, many observation series.

This supersedes the two earlier builders, which between them could not express
the models the R tutorials use. :func:`~epidemia.model.build_model` has a random
walk on ``R_t`` but only one population;
:func:`~epidemia.multilevel.build_multilevel_model` has many regions and
covariates but a deterministic ``R_t`` and a single observed series. Neither can
write ``R(region, date) ~ 1 + rw(time = week, gr = region)`` with deaths and
cases fitted jointly, which is ordinary usage in R.

The model here is the one in ``inst/stan/epidemia_base.stan``:

.. math::

    R^{(m)}_t &= K\\,g^{-1}\\!\\left(b^{(m)}_0
                 + \\sum_k (\\beta_k + b^{(m)}_k) X^{(m)}_{k,t}
                 + w^{(p(m))}_{\\tau(m,t)}\\right) \\\\
    i'^{(m)}_t &= R^{(m)}_t \\sum_{s<t} i^{(m)}_s \\pi_{t-s} \\\\
    i^{(m)}_t  &= S^{(m)}_t\\left(1 - e^{-i'^{(m)}_t / P_m}\\right)
                  \\quad\\text{(population adjustment; otherwise } i = i')\\\\
    \\mathbb{E}\\left[Y^{(s,m)}_t\\right]
       &= \\alpha^{(s,m)}_t \\sum_{k\\ge 1} \\pi^{(s)}_k\\, i^{(m)}_{t-k}

with region effects either independent (R's ``||``) or correlated through an
LKJ-Cholesky covariance (R's ``|`` with ``decov``).

Both kernels are lag-1-first, as in R's Stan: ``gen[0]`` weights yesterday's
infections and an infection is never observed on the day it happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ObsModel",
    "RandomWalk",
    "PanelData",
    "EpiModelConfig",
    "build_epidemia_model",
    "fit_epidemia",
    "prepare_panel",
]


@dataclass
class RandomWalk:
    """A random walk on the linear predictor, mirroring R's ``rw()``.

    Attributes
    ----------
    index : int array (M, T)
        For each region and day, which step of the walk that day belongs to.
        Repeating an index across days gives a coarser walk (R's
        ``rw(time = week)``); ``arange(T)`` broadcast over regions gives a daily
        one. Padded days may carry any in-range value -- they are never used.
    by_region : bool
        ``False`` (default) puts every region on ONE shared walk, which is R's
        ``rw(time = week)``. ``True`` gives each region its own independent walk
        with its own scale, which is R's ``rw(time = week, gr = region)``.
    prior_scale : float
        Scale of the half-normal prior on the walk's step size.
    """

    index: np.ndarray
    by_region: bool = False
    prior_scale: float = 0.2


@dataclass
class ObsModel:
    """One observation series, mirroring R's ``epiobs()``.

    A series is a delayed, partially ascertained view of the latent infections:
    infections are convolved with ``i2o`` and scaled by an ascertainment rate
    that is itself modelled. It is *not* a view of infections directly.

    Attributes
    ----------
    name : str
        Series name, e.g. ``"deaths"``. Used for variable names in the trace.
    y : array (M, T)
        Observed counts, padded. Entries where ``mask`` is False are ignored.
    mask : bool array (M, T)
        Days that enter this series' likelihood. Series may be observed on
        different days -- deaths daily, a survey weekly -- which is why the mask
        belongs to the series and not to the panel.
    i2o : array (K,)
        Infection-to-observation delay distribution, lag-1-first. Should sum to
        one; the ascertainment cap is supplied separately through ``link_K``
        rather than folded in here (R folds it into ``i2o``, which is why R warns
        that "i2o does not sum to one" for a ``scaled_logit`` link).
    family : {"poisson", "neg_binom", "quasi_poisson", "normal", "log_normal"}
        Observation family.
    link_K : float
        Upper bound of the ascertainment rate: ``rate = link_K * sigmoid(eta)``.
        For deaths this is the maximum IFR (R's ``scaled_logit(0.02)``).
    X : array (M, T, Ks) | None
        Covariates for the ascertainment regression. ``None`` for an
        intercept-only rate (R's ``deaths ~ 1``).
    offset : array (M, T) | None
        Optional known additive term on the linear predictor.
    intercept : bool
        Include an intercept in the ascertainment regression. Set ``False`` with
        a full-rank ``X`` to get R's ``~ 0 + region``.
    prior_intercept, prior, prior_aux : Prior | None
        Prior specifications from :mod:`epidemia.priors`, mirroring R's
        ``epiobs(prior_intercept =, prior =, prior_aux =)``. ``None`` keeps the
        scalar-hyperparameter defaults below, so existing code is unaffected.
    prior_intercept_scale, prior_coef_scale : float
        Normal prior scales for the ascertainment intercept and coefficients,
        used when the corresponding ``prior_*`` object is ``None``.
    prior_aux_loc, prior_aux_scale : float
        Auxiliary parameter prior. For ``neg_binom`` the reciprocal dispersion is
        ``loc + scale * HalfNormal(1)``, matching R's ``normal(10, 5)`` truncated
        to be positive.
    """

    name: str
    y: np.ndarray
    mask: np.ndarray
    i2o: np.ndarray
    family: str = "neg_binom"
    link_K: float = 1.0
    X: np.ndarray | None = None
    offset: np.ndarray | None = None
    intercept: bool = True
    prior_intercept: object = None
    prior: object = None
    prior_aux: object = None
    prior_intercept_scale: float = 0.2
    prior_coef_scale: float = 0.5
    prior_aux_loc: float = 10.0
    prior_aux_scale: float = 5.0


@dataclass
class PanelData:
    """Padded, model-ready arrays shared by every observation series.

    Attributes
    ----------
    X : float array (M, T, K)
        Covariates for the ``R_t`` regression. ``K`` may be 0.
    lengths : int array (M,)
        Genuine (modelled) days per region; the rest of each row is padding.
    regions : list[str]
    npis : list[str]
    dates : list[numpy.ndarray]
    pops : array (M,) | None
        Region populations. Required for the susceptibility adjustment.
    """

    X: np.ndarray
    lengths: np.ndarray
    regions: list
    npis: list
    dates: list
    pops: np.ndarray | None = None


@dataclass
class EpiModelConfig:
    """Configuration for :func:`build_epidemia_model`.

    Attributes
    ----------
    gen : array (L,)
        Generation-interval kernel, lag-1-first.
    prior_covariates, prior_intercept, prior_seeds, prior_aux : Prior | None
        Prior specifications from :mod:`epidemia.priors`, mirroring R's
        ``epirt(prior =, prior_intercept =)`` and
        ``epiinf(prior_seeds =, prior_aux =)``. ``None`` keeps the
        scalar-hyperparameter defaults documented below, so existing code is
        unaffected. ``prior_covariates`` replaces the shifted-gamma built from
        ``beta_shape``/``beta_scale``/``beta_shift``; ``prior_aux`` replaces the
        latent-infection dispersion built from ``latent_aux_*``.
    R_link_K : float
        Carrying capacity of the scaled-logit link on ``R_t``.
    intercept : bool
        Include a global intercept in the ``R_t`` regression. R's ``~ 0 + (...)``
        form drops it and lets the region intercepts carry the baseline.
    beta_shape, beta_scale, beta_shift : float
        Shifted-gamma prior on the fixed covariate effects:
        ``beta = shift - Gamma(shape, scale)``, so effects are a priori
        non-positive up to a small shift. R's ``shifted_gamma``.
    correlated : bool
        ``False`` (default) makes the region intercept and slopes independent,
        which is R's ``(1 + x || region)``. ``True`` estimates their full
        covariance through an LKJ-Cholesky prior, which is R's ``(1 + x | region)``
        with ``decov``.
    lkj_eta : float
        LKJ regularization when ``correlated`` is True. 1 is uniform over
        correlation matrices; larger values shrink towards the identity.
    sd_intercept_shape, sd_slope_shape, sd_scale : float
        Gamma priors on the between-region standard deviations.
    sd_slope_fixed : object
        Hold the slope SDs fixed instead of estimating them, selecting the
        no-pooling / full-pooling regimes. Ignored when ``correlated``.
    rw : RandomWalk | None
        Optional random walk on the ``R_t`` linear predictor.
    seed_days : int
    seed_pooling : bool
        Partially pool the seeded infections through a shared mean (R's ``hexp``).
    seed_aux_rate, seed_prior_mean : float
    latent : bool
        Treat infections as parameters with noise around the renewal equation,
        rather than as a deterministic function of it -- R's
        ``epiinf(latent = TRUE)``. Useful when counts are low enough that the
        deterministic recursion is too rigid.
    latent_aux_loc, latent_aux_scale : float
        Prior on the infection dispersion: ``loc + scale * HalfNormal(1)``,
        matching R's ``epiinf(prior_aux = normal(10, 5))`` truncated positive.
    fixed_vtm : bool
        With ``True`` (R's default) the auxiliary parameter is a
        variance-to-mean ratio, so ``sd = sqrt(aux * mean)``. With ``False`` it
        is a coefficient of variation, so ``sd = aux * mean``.
    pop_adjust : bool
        Deplete the susceptible population as infections accumulate. Requires
        ``PanelData.pops``. Without it, infections grow without bound and long
        forecasts are meaningless.
    prior_susc_mean, prior_susc_sd : float
        Prior on the initially susceptible *proportion*. Only used when
        ``pop_adjust``; the default fixes it at 1 (everyone susceptible).
    """

    gen: np.ndarray
    prior_covariates: object = None
    prior_intercept: object = None
    prior_seeds: object = None
    prior_aux: object = None
    R_link_K: float = 6.5
    intercept: bool = False
    beta_shape: float = 1.0 / 6.0
    beta_scale: float = 1.0
    beta_shift: float = float(np.log(1.05) / 6.0)
    correlated: bool = False
    lkj_eta: float = 1.0
    sd_intercept_shape: float = 2.0
    sd_slope_shape: float = 0.5
    sd_scale: float = 0.25
    sd_slope_fixed: object = None
    rw: RandomWalk | None = None
    seed_days: int = 6
    seed_pooling: bool = True
    seed_aux_rate: float = 0.03
    seed_prior_mean: float = 30.0
    latent: bool = False
    latent_aux_loc: float = 10.0
    latent_aux_scale: float = 5.0
    fixed_vtm: bool = True
    pop_adjust: bool = False
    prior_susc_mean: float | None = None
    prior_susc_sd: float = 0.1
    _extra: dict = field(default_factory=dict, repr=False)


_COUNT_FAMILIES = {"poisson", "neg_binom", "quasi_poisson"}
_FAMILIES = _COUNT_FAMILIES | {"normal", "log_normal"}


def _region_effects(pm, pt, config, M, K):
    """Region intercepts and slopes, independent or correlated.

    Returns ``(b0, b)`` with shapes ``(M,)`` and ``(M, K)``. Both are written
    non-centred -- a standard normal scaled by the between-region SD -- because
    the centred form has the funnel geometry that makes these models diverge.
    """
    if config.correlated:
        if K == 0:
            raise ValueError(
                "correlated=True needs at least one covariate; with an intercept "
                "alone there is no covariance to estimate."
            )
        # decov / lkj: Sigma = diag(sd) Omega diag(sd). PyMC gives us the
        # Cholesky factor directly, which is what the non-centred form wants.
        sd_shapes = np.concatenate(
            [[config.sd_intercept_shape], np.full(K, config.sd_slope_shape)]
        )
        sd_dist = pm.Gamma.dist(alpha=sd_shapes, beta=1.0 / config.sd_scale,
                                shape=K + 1)
        chol, _, _ = pm.LKJCholeskyCov(
            "Sigma_chol", n=K + 1, eta=config.lkj_eta,
            sd_dist=sd_dist, compute_corr=True,
        )
        z_full = pm.Normal("z_full", 0.0, 1.0, shape=(M, K + 1))
        b_full = pm.Deterministic("b_full", z_full @ chol.T)
        b0 = pm.Deterministic("b0", b_full[:, 0], dims="region")
        b = pm.Deterministic("b", b_full[:, 1:], dims=("region", "npi"))
        return b0, b

    # Independent (R's `||`): one SD per term, no covariance.
    if K == 0:
        sd0 = pm.Gamma("sd_intercept", alpha=config.sd_intercept_shape,
                       beta=1.0 / config.sd_scale)
        z0 = pm.Normal("z0", 0.0, 1.0, dims="region")
        return pm.Deterministic("b0", sd0 * z0, dims="region"), None

    if config.sd_slope_fixed is None:
        sd_shapes = np.concatenate(
            [[config.sd_intercept_shape], np.full(K, config.sd_slope_shape)]
        )
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

    z0 = pm.Normal("z0", 0.0, 1.0, dims="region")
    z = pm.Normal("z", 0.0, 1.0, dims=("region", "npi"))
    b0 = pm.Deterministic("b0", sd[0] * z0, dims="region")
    b = pm.Deterministic("b", sd[1:] * z, dims=("region", "npi"))
    return b0, b


def _random_walk(pm, pt, rw: RandomWalk, M, T):
    """The walk contribution to the linear predictor, shape ``(M, T)``.

    One shared walk, or one per region when ``by_region``. Non-centred: unit
    increments scaled by the process SD, then accumulated.
    """
    index = np.asarray(rw.index, dtype=int)
    if index.shape != (M, T):
        raise ValueError(
            f"rw.index must have shape {(M, T)}, got {index.shape}"
        )
    n_steps = int(index.max()) + 1
    n_procs = M if rw.by_region else 1

    scale = pm.HalfNormal("rw_scale", rw.prior_scale, shape=n_procs)
    noise = pm.Normal("rw_noise", 0.0, 1.0, shape=(n_procs, n_steps))
    walk = pt.cumsum(scale[:, None] * noise, axis=1)          # (n_procs, n_steps)
    pm.Deterministic("rw", walk)

    proc = np.arange(M) if rw.by_region else np.zeros(M, dtype=int)
    return walk[proc[:, None], index]                          # (M, T)


def _convolve(pt, infections, kernel, M, T):
    """Causal convolution at lags 1..K, vectorised over regions.

    Time-invariant, so this is a genuine convolution and needs no scan: it is a
    sum of shifted copies. The loop unrolls the kernel at graph-build time.
    Taps beyond the modelled window would reference infections from before the
    window, which are not modelled, so the kernel is capped at ``T``.
    """
    terms = [
        kernel[k - 1] * pt.concatenate(
            [pt.zeros((M, k)), infections[:, : T - k]], axis=1
        )
        for k in range(1, min(len(kernel) + 1, T))
    ]
    return pt.add(*terms) if len(terms) > 1 else terms[0]


def _likelihood(pm, pt, obs: ObsModel, E, aux_name):
    """Attach one series' likelihood on its own observed days."""
    from . import priors as _priors

    def _aux():
        """The series' auxiliary parameter, from a Prior spec or the defaults."""
        if obs.prior_aux is not None:
            return _priors.build(obs.prior_aux, aux_name, positive=True)
        raw = pm.HalfNormal(f"{aux_name}_raw", 1.0)
        return pm.Deterministic(
            aux_name, obs.prior_aux_loc + obs.prior_aux_scale * raw
        )

    idx = np.where(np.asarray(obs.mask).reshape(-1))[0]
    mu = E.reshape((-1,))[idx]
    y = np.asarray(obs.y).reshape(-1)[idx]

    fam = obs.family
    if fam not in _FAMILIES:
        raise ValueError(
            f"unknown family {fam!r} for series {obs.name!r}; "
            f"expected one of {sorted(_FAMILIES)}"
        )

    if fam == "poisson":
        pm.Poisson(obs.name, mu=mu, observed=y.astype(int))
        return

    if fam in ("neg_binom", "quasi_poisson"):
        aux = _aux()
        if fam == "neg_binom":
            pm.NegativeBinomial(obs.name, mu=mu, alpha=aux, observed=y.astype(int))
        else:
            # Quasi-Poisson has no exact likelihood; R models the variance-to-mean
            # ratio through a negative binomial with alpha = mu / (d - 1), which
            # gives Var = d * mu. Guard d > 1 so alpha stays positive.
            d = pt.maximum(aux, 1.0 + 1e-6)
            pm.NegativeBinomial(obs.name, mu=mu, alpha=mu / (d - 1.0),
                                observed=y.astype(int))
        return

    sigma = _aux()
    if fam == "normal":
        pm.Normal(obs.name, mu=mu, sigma=sigma, observed=y.astype(float))
    else:  # log_normal
        pm.LogNormal(obs.name, mu=pt.log(mu), sigma=sigma, observed=y.astype(float))


def build_epidemia_model(data: PanelData, obs_models, config: EpiModelConfig):
    """Build the full renewal model as a :class:`pymc.Model`.

    Parameters
    ----------
    data : PanelData
        Covariates and region metadata. ``M`` regions by ``T`` padded days.
    obs_models : ObsModel | list[ObsModel]
        One or more observation series, fitted jointly. Each brings its own
        delay distribution, family, link and ascertainment regression, exactly
        as R's ``obs = list(deaths, cases)`` does.
    config : EpiModelConfig

    Returns
    -------
    pymc.Model
    """
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    from . import priors as _priors

    if isinstance(obs_models, ObsModel):
        obs_models = [obs_models]
    if not obs_models:
        raise ValueError("at least one observation series is required")
    names = [o.name for o in obs_models]
    if len(set(names)) != len(names):
        raise ValueError(f"observation series names must be unique, got {names}")

    gen = np.asarray(config.gen, dtype=float)
    L = gen.shape[0]
    v = int(config.seed_days)

    X = np.asarray(data.X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"data.X must be (M, T, K); got shape {X.shape}")
    M, T, K = X.shape

    for o in obs_models:
        if np.shape(o.y) != (M, T) or np.shape(o.mask) != (M, T):
            raise ValueError(
                f"series {o.name!r}: y and mask must both be {(M, T)}; "
                f"got {np.shape(o.y)} and {np.shape(o.mask)}"
            )

    if config.pop_adjust and data.pops is None:
        raise ValueError("pop_adjust=True requires PanelData.pops")

    coords = {
        "region": list(data.regions),
        "npi": list(data.npis),
        "region_time": np.arange(T),
    }
    with pm.Model(coords=coords) as model:
        # ---- transmission -------------------------------------------------
        eta = pt.zeros((M, T))

        if config.intercept:
            eta = eta + (
                _priors.build(config.prior_intercept, "intercept")
                if config.prior_intercept is not None
                else pm.Normal("intercept", 0.0, 0.5)
            )

        b0, b = _region_effects(pm, pt, config, M, K)
        eta = eta + b0[:, None]

        if K:
            if config.prior_covariates is not None:
                beta = _priors.build(config.prior_covariates, "beta", shape=K)
            else:
                # the default is R's shifted_gamma: effects a priori non-positive
                g_beta = pm.Gamma("g_beta", alpha=config.beta_shape,
                                  beta=1.0 / config.beta_scale, dims="npi")
                beta = pm.Deterministic("beta", config.beta_shift - g_beta,
                                        dims="npi")
            coef = beta[None, :] + b                       # (M, K)
            eta = eta + (pt.as_tensor_variable(X) * coef[:, None, :]).sum(axis=2)

        if config.rw is not None:
            eta = eta + _random_walk(pm, pt, config.rw, M, T)

        R_unadj = pm.Deterministic(
            "Rt_unadj", config.R_link_K * pt.sigmoid(eta),
            dims=("region", "region_time"),
        )

        # ---- infections ---------------------------------------------------
        if config.prior_seeds is not None:
            # hexp() reproduces the pooled form below; any other family gives
            # each region an independent draw.
            seed = _priors.build(config.prior_seeds, "seed", shape=M,
                                 positive=True)
        elif config.seed_pooling:
            tau = pm.Exponential("seed_tau", config.seed_aux_rate)
            seed_raw = pm.Exponential("seed_raw", 1.0, dims="region")
            seed = pm.Deterministic("seed", tau * seed_raw, dims="region")
        else:
            seed = pm.Exponential("seed", 1.0 / config.seed_prior_mean, dims="region")

        seeds = pt.outer(seed, pt.ones(v))
        buf0 = pt.zeros((M, L))
        buf0 = pt.set_subtensor(buf0[:, : min(v, L)], seed[:, None])

        # Latent infections (R's epiinf(latent = TRUE)): the post-seeding
        # infections become parameters, and the renewal equation supplies their
        # MEAN rather than their value. R declares them as
        # `vector<lower=0> infections_raw` whose only density is the state-space
        # term added below, so the parameter itself is flat on the positive half
        # line; the density is proper because each mean depends only on earlier
        # infections.
        latent = bool(config.latent)
        if latent:
            raw = pm.HalfFlat("infections_raw", shape=(M, T - v))

        if config.pop_adjust:
            pops = pt.as_tensor_variable(np.asarray(data.pops, dtype=float))
            if config.prior_susc_mean is None:
                susc0 = pops                                   # everyone susceptible
            else:
                s0 = pm.TruncatedNormal(
                    "S0", mu=config.prior_susc_mean, sigma=config.prior_susc_sd,
                    lower=0.0, upper=1.0, dims="region",
                )
                susc0 = s0 * pops
            # Seeded infections have already happened by the first modelled day.
            S_init = susc0 - v * seed

            if latent:
                def step(R_t, raw_t, buf, S):
                    i_prime = R_t * pt.dot(buf, gen)
                    # Saturating form, as in R's Stan: a large i_prime can never
                    # infect more people than remain susceptible.
                    E_t = S * (1.0 - pt.exp(-i_prime / pops))
                    return (
                        pt.concatenate([raw_t[:, None], buf[:, :-1]], axis=1),
                        S - raw_t,
                        E_t,
                    )

                seqs = [R_unadj[:, v:].T, raw.T]
            else:
                def step(R_t, buf, S):
                    i_prime = R_t * pt.dot(buf, gen)
                    i_t = S * (1.0 - pt.exp(-i_prime / pops))
                    return (
                        pt.concatenate([i_t[:, None], buf[:, :-1]], axis=1),
                        S - i_t,
                        i_t,
                    )

                seqs = [R_unadj[:, v:].T]

            _, S_seq, E_seq = pytensor.scan(
                fn=step, sequences=seqs,
                outputs_info=[buf0, S_init, None], return_updates=False,
            )
            post = raw if latent else E_seq.T
            infections = pt.concatenate([seeds, post], axis=1)
            # R_t as actually realised: the unadjusted rate scaled by the
            # susceptible fraction at the previous step.
            S_full = pt.concatenate(
                [pt.outer(S_init, pt.ones(v)), S_seq.T], axis=1
            )
            pm.Deterministic("susceptible", S_full,
                             dims=("region", "region_time"))
            R = pm.Deterministic(
                "Rt", R_unadj * S_full / pops[:, None],
                dims=("region", "region_time"),
            )
        else:
            if latent:
                def step(R_t, raw_t, buf):
                    E_t = R_t * pt.dot(buf, gen)
                    return pt.concatenate([raw_t[:, None], buf[:, :-1]], axis=1), E_t

                seqs = [R_unadj[:, v:].T, raw.T]
            else:
                def step(R_t, buf):
                    i_t = R_t * pt.dot(buf, gen)
                    return pt.concatenate([i_t[:, None], buf[:, :-1]], axis=1), i_t

                seqs = [R_unadj[:, v:].T]

            _, E_seq = pytensor.scan(
                fn=step, sequences=seqs,
                outputs_info=[buf0, None], return_updates=False,
            )
            post = raw if latent else E_seq.T
            infections = pt.concatenate([seeds, post], axis=1)
            R = pm.Deterministic("Rt", R_unadj,
                                 dims=("region", "region_time"))

        if latent:
            # The state-space density: infections scatter around the renewal
            # mean. fixed_vtm=True makes `aux` a variance-to-mean ratio
            # (sd = sqrt(aux * mean)), False a coefficient of variation
            # (sd = aux * mean) -- R's two options.
            if config.prior_aux is not None:
                inf_aux = _priors.build(config.prior_aux, "inf_aux",
                                        positive=True)
            else:
                aux_raw = pm.HalfNormal("inf_aux_raw", 1.0)
                inf_aux = pm.Deterministic(
                    "inf_aux",
                    config.latent_aux_loc + config.latent_aux_scale * aux_raw,
                )
            mean = pt.maximum(E_seq.T, 1e-9)
            sd = pt.sqrt(inf_aux * mean) if config.fixed_vtm else inf_aux * mean
            pm.Deterministic("E_infections",
                             pt.concatenate([seeds, mean], axis=1),
                             dims=("region", "region_time"))
            pm.Potential(
                "infections_lp",
                pm.logp(pm.Normal.dist(mu=mean, sigma=sd), raw).sum(),
            )

        pm.Deterministic("infections", infections,
                         dims=("region", "region_time"))

        # ---- observation series -------------------------------------------
        for o in obs_models:
            oeta = pt.zeros((M, T))
            if o.intercept:
                oeta = oeta + (
                    _priors.build(o.prior_intercept, f"{o.name}|intercept")
                    if o.prior_intercept is not None
                    else pm.Normal(f"{o.name}|intercept", 0.0,
                                   o.prior_intercept_scale)
                )
            if o.X is not None:
                oX = np.asarray(o.X, dtype=float)
                if oX.shape[:2] != (M, T):
                    raise ValueError(
                        f"series {o.name!r}: X must be (M, T, Ks); got {oX.shape}"
                    )
                ocoef = (
                    _priors.build(o.prior, f"{o.name}|coef", shape=oX.shape[2])
                    if o.prior is not None
                    else pm.Normal(f"{o.name}|coef", 0.0, o.prior_coef_scale,
                                   shape=oX.shape[2])
                )
                oeta = oeta + (pt.as_tensor_variable(oX) * ocoef[None, None, :]).sum(axis=2)
            if o.offset is not None:
                oeta = oeta + pt.as_tensor_variable(
                    np.asarray(o.offset, dtype=float)
                )

            rate = pm.Deterministic(
                f"{o.name}|rate", o.link_K * pt.sigmoid(oeta),
                dims=("region", "region_time"),
            )
            conv = _convolve(pt, infections, np.asarray(o.i2o, dtype=float), M, T)
            E = pm.Deterministic(
                f"E_{o.name}", rate * conv + 1e-6,
                dims=("region", "region_time"),
            )
            _likelihood(pm, pt, o, E, f"{o.name}|aux")

    return model


def prepare_panel(df, npis=(), responses=("deaths",), group="country",
                  date="date", pop=None, seed_offset=30, threshold_on=None,
                  threshold=10, fit_until=None, rw_by=None):
    """Turn a long panel into :class:`PanelData` plus per-series arrays.

    The multi-series counterpart of
    :func:`epidemia.multilevel.prepare_panel`, which handles a single response.
    Each region's modelled window starts ``seed_offset`` days before its
    cumulative ``threshold_on`` series first exceeds ``threshold``, mirroring the
    rule the R vignettes use.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with one row per region-day.
    npis : sequence[str]
        Covariate columns for the ``R_t`` regression.
    responses : sequence[str]
        Response columns, one per observation series. All of them are windowed
        together, so the series share a time axis; a series that is only
        observed on some days (a weekly survey, say) simply has ``False`` in its
        own mask on the rest.
    pop : str | None
        Column holding each region's population, needed for
        ``EpiModelConfig.pop_adjust``. Taken as the first value per region.
    threshold_on : str | None
        Series used to choose each region's start date. Defaults to the first
        entry of ``responses``.
    fit_until : str | None
        Keep only days strictly before this date.
    rw_by : str | None
        Column giving the random-walk step for each day (e.g. an ISO week). When
        given, the returned :class:`PanelData` carries an ``rw_index`` attribute
        suitable for :class:`RandomWalk`.

    Returns
    -------
    (PanelData, dict[str, dict])
        The panel, and a mapping ``{response: {"y": (M,T), "mask": (M,T)}}``
        ready to hand to :class:`ObsModel`.
    """
    import warnings

    import pandas as pd

    responses = list(responses)
    if not responses:
        raise ValueError("at least one response column is required")
    thresh_col = threshold_on or responses[0]
    cutoff = pd.Timestamp(fit_until) if fit_until is not None else None

    per_region = {}
    for name, g in df.sort_values(date).groupby(group, sort=True):
        g = g.reset_index(drop=True)
        crossings = np.where(g[thresh_col].cumsum().to_numpy() > threshold)[0]
        if len(crossings) == 0:
            continue
        start_row = crossings[0] - seed_offset
        if start_row < 0:
            # A negative .iloc would wrap to the end of the series and silently
            # keep only the last few days; clamp instead and say so.
            warnings.warn(
                f"{name!r}: only {crossings[0]} day(s) before cumulative "
                f"{thresh_col} exceeded {threshold}, fewer than seed_offset="
                f"{seed_offset}; starting at the first available day.",
                stacklevel=2,
            )
        else:
            g = g[g[date] > g[date].iloc[start_row]]
        if cutoff is not None:
            g = g[g[date] < cutoff]
        if len(g):
            per_region[str(name)] = g.reset_index(drop=True)

    if not per_region:
        raise ValueError(
            f"no region ever exceeded a cumulative {thresh_col} of {threshold}"
        )

    regions = list(per_region)
    lengths = np.array([len(per_region[r]) for r in regions])
    M, T, K = len(regions), int(lengths.max()), len(npis)

    X = np.zeros((M, T, K), dtype=float)
    rw_index = np.zeros((M, T), dtype=int) if rw_by else None
    pops = np.zeros(M, dtype=float) if pop else None
    series = {r: {"y": np.zeros((M, T)), "mask": np.zeros((M, T), bool)}
              for r in responses}
    dates = []

    for m, region in enumerate(regions):
        g = per_region[region]
        n = len(g)
        if K:
            X[m, :n, :] = g[list(npis)].to_numpy(dtype=float)
        if pop:
            # .iloc[0] would take a NaN if the first row of the window happens
            # to be missing it, which silently produces a zero/NaN population and
            # a division by zero in the susceptibility adjustment.
            finite = g[pop].dropna()
            if finite.empty:
                raise ValueError(
                    f"region {region!r} has no non-missing {pop!r} value; "
                    "population adjustment needs one"
                )
            pops[m] = float(finite.iloc[0])
        if rw_by:
            codes = pd.factorize(g[rw_by], sort=True)[0]
            rw_index[m, :n] = codes
            rw_index[m, n:] = codes[-1] if n else 0
        for r in responses:
            y = g[r].to_numpy(dtype=float)
            observed = np.isfinite(y) & (y >= 0)   # -1 is R's forecast placeholder
            series[r]["y"][m, :n] = np.nan_to_num(y)
            series[r]["mask"][m, :n] = observed
        dates.append(g[date].to_numpy())

    if pops is not None and not np.all(pops > 0):
        bad = [r for r, p in zip(regions, pops) if not p > 0]
        raise ValueError(f"non-positive population for region(s) {bad}")

    panel = PanelData(X=X, lengths=lengths, regions=regions,
                      npis=list(npis), dates=dates, pops=pops)
    if rw_index is not None:
        panel.rw_index = rw_index
    return panel, series


def fit_epidemia(data: PanelData, obs_models, config: EpiModelConfig,
                 draws=1000, tune=1000, chains=4, seed=0,
                 adaptation="low_rank", backend="numba", target_accept=0.95,
                 progress_bar=True, **kwargs):
    """Build and fit the model with nutpie, returning an ``InferenceData``.

    The counterpart of R's ``epim()`` for :func:`build_epidemia_model`.

    ``adaptation`` defaults to ``"low_rank"`` rather than nutpie's ``"diag"``:
    these posteriors combine a hierarchy with a random walk and, often,
    collinear covariates, leaving a correlated ridge that a diagonal mass matrix
    cannot follow. It fails silently -- bad mixing without divergences.
    Benchmarked, ``"low_rank"`` was both faster and better mixed on every model
    tried; see the documentation's "Performance" page.

    ``target_accept`` defaults to 0.95 rather than 0.8 for the related but
    distinct funnel geometry between the between-region SDs and the non-centred
    region effects.

    Parameters mirror :func:`epidemia.multilevel.fit_multilevel`; extra keyword
    arguments are passed through to ``nutpie.sample``.
    """
    from .multilevel import _compile, _warn_on_divergences

    model = build_epidemia_model(data, obs_models, config)
    idata = _compile(model, backend=backend, draws=draws, tune=tune,
                     chains=chains, seed=seed, adaptation=adaptation,
                     progress_bar=progress_bar, target_accept=target_accept,
                     **kwargs)
    _warn_on_divergences(idata)
    return idata
