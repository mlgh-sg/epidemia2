"""One-call forecasting from a fitted :func:`epidemia.core.build_epidemia_model`.

This is the Python counterpart of R's ``posterior_predict(fm, newdata = ...)``,
``posterior_rt(fm, newdata = ...)`` and ``posterior_infections(fm, newdata = ...)``.
In R those hand the fitted parameter draws to Stan's *standalone generated
quantities* and re-run the renewal recursion over a longer time span. Nothing is
re-fitted, and nothing needs to be: conditional on one draw of the parameters
every latent series in the model -- :math:`R_t`, infections, the susceptible
pool, each series' expected value -- is a **deterministic** function of that
draw. So a forecast is plain forward simulation, done here in NumPy and
vectorised over the draw axis.

:mod:`epidemia.predict` already has the primitives (``simulate``,
``expected_observations``, ``posterior_predict``) but leaves the user to
reconstruct the linear predictor, the design matrices over the extended window
and the per-series bookkeeping by hand. This module does that end to end:

    fc = forecast(idata, panel, obs_models, config, newdata=future_df)
    fc.Rt            # (draws, regions, days)
    fc.predicted["deaths"]

Two conventions are inherited from the R tutorials, which extend their data
frame with future rows whose NPI columns simply repeat the last observed value:

* **covariates are carried forward.** Any day of the extended window without a
  covariate value (missing column entries, or days past the end of ``newdata``)
  reuses the last row that had one.
* **the random walk is held at its final fitted step.** The fitted walk has no
  increments beyond the last week it saw; continuing it would mean *inventing*
  new random increments, which is a different (and much wider) forecast. Holding
  it makes the forecast explicitly "R_t stays where it ended", which is what
  ``epidemia``'s vignettes plot.

Both kernels keep the package-wide lag-1-first convention (see
:mod:`epidemia.renewal`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import EpiModelConfig, ObsModel, PanelData
from .predict import expected_observations, posterior_predict

__all__ = ["Forecast", "forecast"]


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------


@dataclass
class Forecast:
    """Posterior draws of every latent series over the extended window.

    Every array has the draw axis first, so ``arr[:, m, t]`` is the posterior of
    region ``m`` on day ``t`` and quantiles are ``np.percentile(arr, q, axis=0)``.

    Attributes
    ----------
    regions : list[str]
        Region names, in the order of the leading axis of ``lengths`` and of
        axis 1 of every array.
    dates : list[numpy.ndarray]
        Per-region dates of the extended window. ``dates[m]`` has
        ``lengths[m]`` entries.
    lengths : int array (M,)
        Genuine days per region over the extended window. Days beyond this are
        padding: the arrays are rectangular, so a region that ends early is
        carried forward with its last covariate row and its values there are
        meaningless. Trim with ``lengths`` (:meth:`to_frame` does).
    Rt : array (draws, M, T)
        Reproduction number as realised, i.e. after the susceptibility
        adjustment. Identical to ``Rt_unadj`` when ``config.pop_adjust`` is off.
    Rt_unadj : array (draws, M, T)
        ``R_link_K * sigmoid(eta)``, before any depletion of susceptibles.
    infections : array (draws, M, T)
        Latent daily infections, seeded days included.
    susceptible : array (draws, M, T) | None
        Remaining susceptibles; ``None`` unless ``config.pop_adjust``.
    expected : dict[str, array (draws, M, T)]
        Per series, the model's ``E_<series>``: the expected observation.
    predicted : dict[str, array (draws, M, T)]
        Per series, draws from the observation family around ``expected``. These
        carry observation noise as well as parameter uncertainty -- banding
        ``expected`` alone gives intervals far too narrow to compare with data.
    families : dict[str, str]
        The family each series was drawn from, for reference.
    draw_index : int array (draws,)
        Which flattened posterior draws (chain-major) were used. Useful to line a
        forecast up with other per-draw quantities from the same fit.
    """

    regions: list
    dates: list
    lengths: np.ndarray
    Rt: np.ndarray
    Rt_unadj: np.ndarray
    infections: np.ndarray
    susceptible: np.ndarray | None
    expected: dict
    predicted: dict
    families: dict = field(default_factory=dict)
    draw_index: np.ndarray | None = None

    @property
    def n_draws(self) -> int:
        """Number of posterior draws carried by every array."""
        return int(self.Rt.shape[0])

    @property
    def series(self) -> list:
        """Names of the observation series in this forecast."""
        return list(self.expected)

    def to_frame(self, probs=(0.025, 0.25, 0.5, 0.75, 0.975)):
        """Summarise as a tidy long frame, one row per region-day-variable.

        Parameters
        ----------
        probs : sequence[float]
            Quantiles of the posterior to report, as columns named ``q2.5`` etc.

        Returns
        -------
        pandas.DataFrame
            Columns ``region``, ``date``, ``variable``, ``mean`` and one column
            per entry of ``probs``. Padding days are dropped.
        """
        import pandas as pd

        probs = list(probs)
        blocks = {"Rt": self.Rt, "Rt_unadj": self.Rt_unadj,
                  "infections": self.infections}
        if self.susceptible is not None:
            blocks["susceptible"] = self.susceptible
        for name, arr in self.expected.items():
            blocks[f"E_{name}"] = arr
        for name, arr in self.predicted.items():
            blocks[name] = arr

        frames = []
        for variable, arr in blocks.items():
            arr = np.asarray(arr, dtype=float)
            mean = arr.mean(axis=0)
            qs = np.percentile(arr, [100.0 * p for p in probs], axis=0)
            for m, region in enumerate(self.regions):
                n = int(self.lengths[m])
                out = {"region": region, "date": np.asarray(self.dates[m])[:n],
                       "variable": variable, "mean": mean[m, :n]}
                for j, p in enumerate(probs):
                    out[f"q{100 * p:g}"] = qs[j, m, :n]
                frames.append(pd.DataFrame(out))
        return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _sigmoid(x):
    """Logistic function, written through ``tanh`` so large |x| cannot overflow."""
    return 0.5 * (np.tanh(0.5 * np.asarray(x, dtype=float)) + 1.0)


def _stack_draws(post, name, required=True):
    """Pull one posterior variable out as ``(n_samples, ...)``, chain-major.

    Chains are folded into the leading axis in the same order everywhere in this
    module, so ``draw_index`` means the same thing for every variable.
    """
    if name not in post:
        if not required:
            return None
        raise KeyError(
            f"posterior has no variable {name!r}; forecast() needs the "
            "deterministics that build_epidemia_model records "
            "(b0, b, beta, rw, seed, '<series>|rate' inputs, '<series>|aux'). "
            f"Available: {sorted(post.data_vars)}"
        )
    da = post[name]
    rest = [d for d in da.dims if d not in ("chain", "draw")]
    arr = da.transpose("chain", "draw", *rest).values
    return arr.reshape((-1,) + arr.shape[2:])


def _carry_forward(arr, lengths, T_ext):
    """Extend a fitted ``(M, T, ...)`` design array to ``(M, T_ext, ...)``.

    Fitted days are copied verbatim (so an in-sample forecast reproduces the fit
    exactly, padding included); days past the fit repeat each region's last
    *genuine* row, which is the "carry the last observed row forward" rule of the
    R tutorials.
    """
    arr = np.asarray(arr)
    M, T = arr.shape[0], arr.shape[1]
    if T_ext <= T:
        return arr[:, :T_ext]
    out = np.empty((M, T_ext) + arr.shape[2:], dtype=arr.dtype)
    out[:, :T] = arr
    for m in range(M):
        last = max(int(lengths[m]) - 1, 0)
        out[m, T:] = arr[m, last]
    return out


def _forward_fill(x):
    """Carry the last finite row of a ``(n, K)`` array forward over NaN rows.

    Leading NaNs have nothing to carry from, so they are left as-is and the
    caller decides (here: they are an error, since the fitted window must be
    covered).
    """
    x = np.array(x, dtype=float, copy=True)
    if x.ndim == 1:
        x = x[:, None]
    for t in range(1, x.shape[0]):
        bad = ~np.isfinite(x[t])
        if bad.any():
            x[t, bad] = x[t - 1, bad]
    return x


# --------------------------------------------------------------------------
# design over the extended window
# --------------------------------------------------------------------------


def _design_from_newdata(panel: PanelData, newdata, group, date):
    """Rebuild the ``R_t`` covariates over the longer window ``newdata`` spans.

    Returns ``(X_ext, dates_ext, n_ext)``. Each region is taken from its own
    fitted start date onwards, so day ``t`` of the output is day ``t`` of the
    fit for as long as the fit lasted -- that alignment is what makes the first
    rows of a forecast equal the in-sample values.
    """
    import pandas as pd

    if group not in newdata.columns:
        raise ValueError(
            f"newdata has no column {group!r}; pass group= to name the region "
            f"column (columns: {list(newdata.columns)})"
        )
    if date not in newdata.columns:
        raise ValueError(f"newdata has no column {date!r}; pass date=")
    missing = [c for c in panel.npis if c not in newdata.columns]
    if missing:
        raise ValueError(f"newdata is missing covariate column(s) {missing}")

    nd = newdata.copy()
    col = nd[date]
    # Strings/objects compare lexicographically against the fitted dates, which
    # silently mis-windows; coerce anything non-numeric to datetimes.
    if not (np.issubdtype(col.dtype, np.number)
            or np.issubdtype(col.dtype, np.datetime64)):
        nd[date] = pd.to_datetime(col)
    keys = nd[group].astype(str)

    K = len(panel.npis)
    X_rows, dates_ext, n_ext = [], [], []
    for m, region in enumerate(panel.regions):
        g = nd[keys == str(region)].sort_values(date)
        if g.empty:
            raise ValueError(f"newdata has no rows for region {region!r}")
        start = np.asarray(panel.dates[m])[0]
        d = g[date].to_numpy()
        g = g.loc[d >= start]
        if g.empty or g[date].to_numpy()[0] != start:
            raise ValueError(
                f"newdata for region {region!r} does not start at the fitted "
                f"start date {start!r}; a forecast extends the fitted window, "
                "so newdata must cover all of it."
            )
        n = len(g)
        if n < int(panel.lengths[m]):
            raise ValueError(
                f"newdata for region {region!r} has {n} day(s) from the fitted "
                f"start but the fit used {int(panel.lengths[m])}"
            )
        if K:
            x = _forward_fill(g[list(panel.npis)].to_numpy(dtype=float))
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"region {region!r}: covariate(s) missing on the first day "
                    "of the window, with nothing earlier to carry forward"
                )
        else:
            x = np.zeros((n, 0))
        X_rows.append(x)
        dates_ext.append(g[date].to_numpy())
        n_ext.append(n)

    n_ext = np.asarray(n_ext, dtype=int)
    T_ext = max(int(n_ext.max()), int(np.asarray(panel.X).shape[1]))
    X = np.zeros((len(panel.regions), T_ext, K), dtype=float)
    for m, x in enumerate(X_rows):
        n = n_ext[m]
        X[m, :n] = x
        if n < T_ext and n:
            X[m, n:] = x[-1]          # nothing more is known: hold the last row
    return X, dates_ext, n_ext, T_ext


# --------------------------------------------------------------------------
# the recursion
# --------------------------------------------------------------------------


def _renewal(Rt_unadj, gen, seed, seed_days, pops=None, susc0=None):
    """Renewal recursion over ``(draws, regions, days)``.

    A transcription of the two ``pytensor.scan`` branches of
    :func:`epidemia.core.build_epidemia_model`, kept separate from
    :func:`epidemia.predict.simulate` on purpose: that function follows R's
    standalone Stan code, which adjusts the *seeding* days for susceptibility
    too and starts the pool at ``pops``. ``build_epidemia_model`` instead holds
    the seeded days at ``seed`` exactly and starts the pool at
    ``susc0 - seed_days * seed``. Forecasts have to match the model that was
    fitted, or the in-sample part of the forecast would not reproduce the fit.

    Returns ``(infections, susceptible)``; ``susceptible`` is ``None`` when
    ``pops`` is ``None``.
    """
    gen = np.asarray(gen, dtype=float)
    S, M, T = Rt_unadj.shape
    v, L = int(seed_days), gen.shape[0]

    R = Rt_unadj.reshape(-1, T)
    sd = np.broadcast_to(seed, (S, M)).reshape(-1)
    inf = np.zeros((S * M, T))
    inf[:, :min(v, T)] = sd[:, None]

    if pops is None:
        for t in range(v, T):
            lo = max(0, t - L)
            window = inf[:, lo:t][:, ::-1]        # i_{t-1}, i_{t-2}, ...
            inf[:, t] = R[:, t] * (window @ gen[: t - lo])
        return inf.reshape(S, M, T), None

    p = np.broadcast_to(np.asarray(pops, dtype=float), (S, M)).reshape(-1)
    if np.any(p <= 0):
        raise ValueError("populations must be positive for pop_adjust=True")
    s0 = p if susc0 is None else np.broadcast_to(susc0, (S, M)).reshape(-1)
    state = np.asarray(s0, dtype=float) - v * sd   # seeds already happened
    susc = np.empty((S * M, T))
    susc[:, :min(v, T)] = state[:, None]

    for t in range(v, T):
        lo = max(0, t - L)
        window = inf[:, lo:t][:, ::-1]
        i_prime = R[:, t] * (window @ gen[: t - lo])
        # -expm1(-x) == 1 - exp(-x) but accurate for the tiny x of a big pop.
        i_t = state * -np.expm1(-i_prime / p)
        inf[:, t] = i_t
        state = state - i_t                        # i_t <= state, so state >= 0
        susc[:, t] = state

    return inf.reshape(S, M, T), susc.reshape(S, M, T)


def _draw_series(E, obs: ObsModel, aux, rng):
    """Draw from one series' family, matching how ``core`` parameterises it.

    ``core`` and :func:`epidemia.predict.posterior_predict` do not use the same
    auxiliary parameterisation for two families, so translate rather than pass
    ``aux`` through blindly:

    * ``quasi_poisson``: ``core`` fits ``alpha = mu / (d - 1)``, i.e. ``aux`` is
      the variance-to-mean ratio ``d``; ``posterior_predict`` takes ``mu / aux``.
    * ``log_normal``: ``core`` fits ``LogNormal(mu = log(E), sigma)``, whose mean
      is ``E * exp(sigma^2 / 2)``; ``posterior_predict`` takes a location whose
      exponential is the mean.
    """
    fam = obs.family
    if fam == "poisson":
        return posterior_predict(E, "poisson", rng=rng)

    if aux is None:
        raise KeyError(
            f"series {obs.name!r} has family {fam!r} but the posterior has no "
            f"'{obs.name}|aux'"
        )
    a = np.asarray(aux, dtype=float).reshape((-1, 1, 1))

    if fam == "quasi_poisson":
        d = np.maximum(a, 1.0 + 1e-6)              # exactly core's guard
        return posterior_predict(E, "quasi_poisson", aux=d - 1.0, rng=rng)
    if fam == "log_normal":
        return posterior_predict(np.log(E) + a**2 / 2.0, "log_normal",
                                 aux=a, rng=rng)
    return posterior_predict(E, fam, aux=a, rng=rng)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def forecast(idata, panel: PanelData, obs_models, config: EpiModelConfig,
             newdata=None, draws=None, seed=None, series=None,
             group="country", date="date"):
    """Forecast every latent and observed series from a fitted model.

    Rebuilds the model's design over a (possibly longer) window, reconstructs
    the linear predictor from the posterior draws exactly as
    :func:`epidemia.core.build_epidemia_model` does::

        eta        = b0[m] + sum_k (beta_k + b[m, k]) X[m, t, k] + rw[p(m), tau(m, t)]
        Rt_unadj   = config.R_link_K * sigmoid(eta)

    then runs the renewal recursion (with the susceptibility adjustment when
    ``config.pop_adjust``) and, per observation series::

        rate = link_K * sigmoid(obs eta)
        E    = rate * conv(infections, i2o) + 1e-6

    finally drawing from that series' family. Nothing is re-fitted.

    Parameters
    ----------
    idata : arviz.InferenceData
        The fit, with the ``posterior`` group produced by
        :func:`epidemia.core.fit_epidemia`.
    panel : PanelData
        The panel the model was fitted to. Its ``dates``, ``lengths``, ``npis``
        and ``pops`` define the fitted window and the alignment of ``newdata``.
    obs_models : ObsModel | list[ObsModel]
        The same observation models that were fitted.
    config : EpiModelConfig
        The same configuration that was fitted.
    newdata : pandas.DataFrame, optional
        A long frame like the one :func:`epidemia.core.prepare_panel` consumed,
        covering a longer period. Days past the fit take their covariates from
        here; missing values (and days past the end of ``newdata`` for a region
        that stops early) carry the last known row forward. ``None`` forecasts
        the fitted window only, which is the in-sample fit.
    draws : int, optional
        Subsample this many posterior draws (without replacement). The cost and
        the memory of a forecast are linear in the number of draws, so a few
        hundred is usually plenty for plotting.
    seed : int | numpy.random.Generator, optional
        Seeds both the subsampling and the predictive draws, so the same seed
        reproduces the whole forecast.
    series : str | sequence[str], optional
        Restrict the observation series to forecast. Defaults to all of them.
    group, date : str
        Region and date columns of ``newdata``; named as in
        :func:`epidemia.core.prepare_panel`.

    Returns
    -------
    Forecast

    Notes
    -----
    The random walk is **held at each region's final fitted step** over the
    forecast horizon (see the module docstring), so with covariates also carried
    forward the forecast says "R_t stays where it ended".
    """
    if isinstance(obs_models, ObsModel):
        obs_models = [obs_models]
    obs_models = list(obs_models)
    if series is not None:
        wanted = [series] if isinstance(series, str) else list(series)
        known = {o.name for o in obs_models}
        bad = [s for s in wanted if s not in known]
        if bad:
            raise ValueError(f"unknown series {bad}; have {sorted(known)}")
        obs_models = [o for o in obs_models if o.name in wanted]

    if not hasattr(idata, "posterior"):
        raise ValueError("idata has no 'posterior' group")
    post = idata.posterior

    X_fit = np.asarray(panel.X, dtype=float)
    M, T_fit, K = X_fit.shape
    lengths = np.asarray(panel.lengths, dtype=int)

    # ---- design over the extended window --------------------------------
    if newdata is None:
        X_ext = X_fit
        T_ext = T_fit
        n_ext = lengths.copy()
        dates_ext = [np.asarray(d) for d in panel.dates]
    else:
        X_ext, dates_ext, n_ext, T_ext = _design_from_newdata(
            panel, newdata, group, date
        )

    # ---- posterior draws -------------------------------------------------
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    b0_all = _stack_draws(post, "b0")
    n_samples = b0_all.shape[0]
    if draws is not None and int(draws) < n_samples:
        idx = np.sort(rng.choice(n_samples, size=int(draws), replace=False))
    else:
        idx = np.arange(n_samples)
    S = idx.shape[0]

    def take(name, required=True):
        arr = _stack_draws(post, name, required=required)
        return None if arr is None else np.asarray(arr, dtype=float)[idx]

    b0 = np.asarray(b0_all, dtype=float)[idx]                   # (S, M)

    eta = np.zeros((S, M, T_ext))
    if config.intercept:
        eta += take("intercept").reshape(S, 1, 1)
    eta += b0[:, :, None]

    if K:
        beta = take("beta").reshape(S, 1, K)
        b = take("b").reshape(S, M, K)
        coef = beta + b                                          # (S, M, K)
        eta += np.einsum("smk,mtk->smt", coef, X_ext)

    if config.rw is not None:
        walk = take("rw")                                        # (S, P, n_steps)
        if walk.ndim == 2:                                       # single process
            walk = walk[:, None, :]
        index = np.asarray(config.rw.index, dtype=int)
        if index.shape != (M, T_fit):
            raise ValueError(
                f"config.rw.index must be {(M, T_fit)}, got {index.shape}"
            )
        # Hold each region's walk at its last *genuine* fitted step. Beyond the
        # fit there are simply no increments to use.
        held = np.minimum(np.arange(T_ext)[None, :], (lengths - 1)[:, None])
        idx_ext = index[np.arange(M)[:, None], held]             # (M, T_ext)
        if idx_ext.max() >= walk.shape[2]:
            raise ValueError(
                f"rw index reaches step {idx_ext.max()} but the posterior walk "
                f"has {walk.shape[2]} step(s)"
            )
        proc = np.arange(M) if config.rw.by_region else np.zeros(M, dtype=int)
        if config.rw.by_region and walk.shape[1] != M:
            raise ValueError(
                f"rw.by_region=True needs {M} walks, posterior has {walk.shape[1]}"
            )
        eta += walk[:, proc[:, None], idx_ext]

    Rt_unadj = float(config.R_link_K) * _sigmoid(eta)

    # ---- infections ------------------------------------------------------
    seed_draws = take("seed")                                    # (S, M)
    if config.pop_adjust:
        if panel.pops is None:
            raise ValueError("config.pop_adjust=True requires PanelData.pops")
        pops = np.asarray(panel.pops, dtype=float)
        s0 = take("S0", required=False)
        susc0 = pops[None, :] if s0 is None else s0 * pops[None, :]
        infections, susceptible = _renewal(
            Rt_unadj, config.gen, seed_draws, config.seed_days,
            pops=pops[None, :], susc0=susc0,
        )
        Rt = Rt_unadj * susceptible / pops[None, :, None]
    else:
        infections, susceptible = _renewal(
            Rt_unadj, config.gen, seed_draws, config.seed_days
        )
        Rt = Rt_unadj

    # ---- observation series ---------------------------------------------
    expected, predicted, families = {}, {}, {}
    for o in obs_models:
        oeta = np.zeros((S, M, T_ext))
        if o.intercept:
            oeta += take(f"{o.name}|intercept").reshape(S, 1, 1)
        if o.X is not None:
            oX = _carry_forward(np.asarray(o.X, dtype=float), lengths, T_ext)
            ocoef = take(f"{o.name}|coef").reshape(S, oX.shape[2])
            oeta += np.einsum("sk,mtk->smt", ocoef, oX)
        if o.offset is not None:
            oeta += _carry_forward(np.asarray(o.offset, dtype=float),
                                   lengths, T_ext)[None]

        rate = float(o.link_K) * _sigmoid(oeta)
        # 1e-6 floor, exactly as build_epidemia_model records E_<series>.
        E = expected_observations(infections, np.asarray(o.i2o, dtype=float),
                                  rate) + 1e-6
        aux = take(f"{o.name}|aux", required=False)
        expected[o.name] = E
        predicted[o.name] = _draw_series(E, o, aux, rng)
        families[o.name] = o.family

    return Forecast(
        regions=list(panel.regions), dates=dates_ext,
        lengths=np.asarray(n_ext, dtype=int),
        Rt=Rt, Rt_unadj=Rt_unadj, infections=infections,
        susceptible=susceptible, expected=expected, predicted=predicted,
        families=families, draw_index=idx,
    )
