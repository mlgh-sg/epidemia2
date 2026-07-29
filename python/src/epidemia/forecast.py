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
* **a latent fit is continued stochastically.** With ``epiinf(latent = TRUE)``
  the fitted window's infections are read back from the posterior and the
  horizon is drawn from ``Normal(mean, sd(mean))`` truncated at zero, which is
  R's ``normal_lb_rng(mu, sigma, 0)``. Running the deterministic recursion
  instead would be a different model from the one that was fitted.
* **the random walk keeps walking.** Past the fitted window new increments are
  drawn at the walk's own fitted scale and cumulated, so forecast ``R_t`` fans
  out -- this is what R does (``new_rw_stanmat``, ``R/pp_eta.R:118-140``). Pass
  ``rw_forecast="hold"`` to freeze the walk at its last fitted step instead,
  which gives a narrower interval and a flat median ``R_t``.

Both kernels keep the package-wide lag-1-first convention (see
:mod:`epidemia.renewal`).

**Latent infections cannot be forecast.** With ``config.latent`` (R's
``epiinf(latent = TRUE)``) the post-seeding infections are *free parameters* --
:func:`epidemia.core.build_epidemia_model` declares them ``HalfFlat`` and the
renewal equation only supplies their mean through a state-space potential. There
is therefore no rule that continues them past the last fitted day, and running
the deterministic recursion instead would quietly return a *different* model's
forecast: plausible-looking, narrower, and not the posterior that was fitted. So
:func:`forecast` refuses to extend a latent fit. The fitted window itself is
still available (``newdata=None``), because there the infections are recorded in
the posterior and are simply read back.

Forecasts also score. :meth:`Forecast.score` lines the predictive draws up with
the observed counts of the matching :class:`~epidemia.core.ObsModel`, drops the
padding, the unmodelled days and the ``-1`` forecast placeholders, and hands the
result to :func:`epidemia.scoring.evaluate_forecast`::

    fc.score(obs_models).error       # group/date/series + CRPS, MAE, ...
    fc.score(obs_models).coverage    # empirical coverage of each CI level
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

    def score(self, obs_models, series=None, levels=(50, 95), groups=None,
              metrics=None):
        """Score the forecast against the observations -- R's ``evaluate_forecast``.

        :mod:`epidemia.scoring` is deliberately model-agnostic: it wants ``y`` as
        ``(N,)`` and ``draws`` as ``(N, S)``. A :class:`Forecast` holds
        ``(draws, region, time)`` arrays with padding, so composing the two by
        hand means transposing, trimming and relabelling. This does that.

        Days that the series never observed, padded days, and R's ``-1``
        forecast placeholders are all dropped, as :mod:`epidemia.scoring` does.

        Parameters
        ----------
        obs_models : ObsModel | list[ObsModel]
            The series definitions the fit used; supplies the observed values.
        series : str | None
            Which series to score. Required when there is more than one.
        levels : sequence[float]
            Credible levels for the coverage table.
        groups : sequence[str] | None
            Restrict to these regions. Unknown names are an error, as in R's
            ``gr_subset``, rather than being silently skipped.
        metrics : sequence[str] | str | None
            Which error metrics to compute, as R's ``evaluate_forecast(metrics=)``.
            ``None`` computes all of them.

        Returns
        -------
        epidemia.scoring.ForecastEvaluation
        """
        import numpy as _np

        from .core import ObsModel as _ObsModel
        from .scoring import evaluate_forecast as _evaluate

        if isinstance(obs_models, _ObsModel):
            obs_models = [obs_models]
        names = [o.name for o in obs_models]

        if series is None:
            scorable = [n for n in names if n in self.predicted]
            if len(scorable) != 1:
                raise ValueError(
                    f"this forecast has {len(scorable)} series ({', '.join(scorable)}); "
                    "pass series= to say which to score"
                )
            series = scorable[0]
        if series not in self.predicted:
            raise ValueError(
                f"no series named {series!r} in the forecast; "
                f"have {sorted(self.predicted)}"
            )
        obs = next((o for o in obs_models if o.name == series), None)
        if obs is None:
            raise ValueError(f"no ObsModel named {series!r} in obs_models")

        draws = self.predicted[series]                       # (S, M, T)
        y_all = _np.asarray(obs.y, dtype=float)
        mask_all = _np.asarray(obs.mask, dtype=bool)

        if groups is not None:
            groups = [str(g) for g in _np.atleast_1d(_np.asarray(groups, dtype=object))]
            unknown = set(groups) - set(map(str, self.regions))
            if unknown:
                # R's gr_subset errors on an unknown group. Skipping silently
                # turns a typo into a quietly narrower score, and an all-bad
                # list into the unrelated "no observed days to score".
                raise ValueError(
                    f"group(s) {sorted(unknown)} not found; have "
                    f"{sorted(map(str, self.regions))}"
                )

        ys, cols, regs, dates = [], [], [], []
        for m, region in enumerate(self.regions):
            if groups is not None and str(region) not in groups:
                continue
            n = int(self.lengths[m])
            # The fit's mask only covers the FITTED window; anything beyond it
            # is out of sample and has no observation to score against.
            n_obs = min(n, mask_all.shape[1])
            keep = _np.where(mask_all[m, :n_obs])[0]
            if not len(keep):
                continue
            ys.append(y_all[m, keep])
            cols.append(draws[:, m, keep])
            regs.extend([region] * len(keep))
            dates.extend(_np.asarray(self.dates[m])[keep])

        if not ys:
            raise ValueError("no observed days to score")

        y = _np.concatenate(ys)
        d = _np.concatenate(cols, axis=1).T                  # (N, S)
        return _evaluate(y, d, group=_np.asarray(regs),
                         date=_np.asarray(dates), levels=levels,
                         metrics=metrics)

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


def _apply_link(link, eta, K):
    """Inverse-link ``eta``. Mirrors :func:`epidemia.core._apply_link`.

    Kept in step with core by the test that reproduces the model's own
    E_<series> from a forecast over the fitted window: if the two drift apart,
    that test fails.
    """
    import math

    if link == "log":
        return np.exp(eta)
    if link == "identity":
        return eta
    if link == "scaled_logit":
        return K * _sigmoid(eta)
    if link == "logit":
        return _sigmoid(eta)
    if link == "probit":
        from scipy.special import ndtr

        return ndtr(eta)
    if link == "cauchit":
        return 0.5 + np.arctan(eta) / math.pi
    if link == "cloglog":
        return 1.0 - np.exp(-np.exp(eta))
    raise ValueError(f"unknown link {link!r}")


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


def _renewal(Rt_unadj, gen, seed, seed_days, pops=None, susc0=None, rm=None,
             latent_sd=None, rng=None, observed=None):
    """Renewal recursion over ``(draws, regions, days)``.

    A transcription of the two ``pytensor.scan`` branches of
    :func:`epidemia.core.build_epidemia_model`, kept separate from
    :func:`epidemia.predict.simulate` on purpose: that function follows R's
    standalone generated-quantities block, which is written against a different
    indexing convention. This one must match ``build_epidemia_model`` exactly,
    or the in-sample part of a forecast would not reproduce the fit -- which is
    what the test of that name checks.

    ``latent_sd`` turns the recursion stochastic, for ``epiinf(latent = TRUE)``:
    each day's infections are drawn from ``Normal(mean, sd(mean))`` truncated at
    zero rather than set to the mean, which is R's ``normal_lb_rng(mu, sigma, 0)``
    in ``gen_infections_pp.stan``. ``observed`` supplies the fitted window's
    infections so they are read back from the posterior instead of re-simulated,
    matching R's ``-0.5`` sentinel for "not sampled".

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

    obs_flat = None if observed is None else np.asarray(observed).reshape(S * M, -1)
    n_obs = 0 if obs_flat is None else obs_flat.shape[1]
    if n_obs:
        inf[:, :min(n_obs, T)] = obs_flat[:, :min(n_obs, T)]

    def draw(mean, t):
        """Mean, or a truncated-normal draw around it for a latent fit."""
        if latent_sd is None:
            return mean
        s_t = latent_sd(np.maximum(mean, 1e-9))
        out = rng.normal(mean, s_t)
        return np.maximum(out, 0.0)               # R's lower bound of 0

    if pops is None:
        for t in range(v, T):
            if t < n_obs:
                continue                          # inside the fit: read back
            lo = max(0, t - L)
            window = inf[:, lo:t][:, ::-1]        # i_{t-1}, i_{t-2}, ...
            inf[:, t] = draw(R[:, t] * (window @ gen[: t - lo]), t)
        return inf.reshape(S, M, T), None

    p = np.broadcast_to(np.asarray(pops, dtype=float), (S, M)).reshape(-1)
    if np.any(p <= 0):
        raise ValueError("populations must be positive for pop_adjust=True")
    s0 = p if susc0 is None else np.broadcast_to(susc0, (S, M)).reshape(-1)
    # Starts at the FULL pool: the seeded infections are saturated and removed by
    # the recursion itself, exactly as core (and R's Stan) do.
    state = np.asarray(s0, dtype=float)
    susc = np.empty((S * M, T))

    if rm is None:
        rm_arr = np.zeros((M, T))
    else:
        rm_arr = np.broadcast_to(np.asarray(rm, dtype=float), (M, T))
    rm_flat = np.repeat(rm_arr[None, :, :], S, axis=0).reshape(S * M, T)

    for t in range(T):
        if t < v:
            pre = sd                                # the seeding window
        else:
            lo = max(0, t - L)
            window = inf[:, lo:t][:, ::-1]
            pre = R[:, t] * (window @ gen[: t - lo])
        # -expm1(-x) == 1 - exp(-x) but accurate for the tiny x of a big pop.
        i_t = state * -np.expm1(-pre / p)
        if t >= n_obs:
            i_t = draw(i_t, t)
        else:
            i_t = inf[:, t]                       # inside the fit: read back
        inf[:, t] = i_t
        state = (1.0 - rm_flat[:, t]) * (state - i_t)
        susc[:, t] = state

    return inf.reshape(S, M, T), susc.reshape(S, M, T)



def _extend_rw_index(index, lengths, T_ext):
    """Continue each region's walk index past its fitted window, same cadence.

    The fitted index says which walk step each day belongs to. Beyond the fitted
    window there are no more entries, so the cadence (days per step -- 7 for a
    weekly walk) is inferred from the fitted rows and continued.
    """
    M, T_fit = index.shape
    out = np.zeros((M, T_ext), dtype=int)
    for m in range(M):
        n = min(int(lengths[m]), T_fit)
        row = index[m, :n]
        keep = min(n, T_ext)
        out[m, :keep] = row[:keep]
        if T_ext <= n:
            continue
        # -1 means "no walk term on this day" and must survive the copy above;
        # the cadence is inferred from the real steps only.
        real = row[row >= 0]
        if real.size == 0:
            out[m, n:] = -1
            continue
        _, counts = np.unique(real, return_counts=True)
        period = max(int(np.median(counts)) if counts.size else 1, 1)
        last_step = int(real[-1])
        spent = int(np.sum(real == last_step))
        step, k = last_step, spent
        for t in range(n, T_ext):
            if k >= period:
                step += 1
                k = 0
            out[m, t] = step
            k += 1
    return out


def _extend_walk(walk, n_needed, scale, rng):
    """Append drawn increments so the walk covers ``n_needed`` steps.

    R's ``new_rw_stanmat`` draws ``rnorm(n) * sigma`` for each period past the
    fitted window and cumulates them (``R/pp_eta.R:118-140``), so a forecast
    ``R_t`` fans out at the walk's own fitted scale. Holding the walk instead
    gives a visibly narrower interval and a flat median.
    """
    S, P, have = walk.shape
    if n_needed <= have:
        return walk
    extra = n_needed - have
    sig = np.asarray(scale, dtype=float)
    if sig.ndim == 1:                       # (S,) -> one scale shared by all walks
        sig = sig[:, None]
    sig = np.broadcast_to(sig.reshape(S, -1), (S, P))[:, :, None]
    steps = rng.normal(size=(S, P, extra)) * sig
    future = walk[:, :, -1:] + np.cumsum(steps, axis=2)
    return np.concatenate([walk, future], axis=2)



def _walk_eta(rw, prefix, take, rng, M, T_fit, T_ext, lengths, rw_forecast):
    """Contribution of every random walk on one predictor, over the extended window.

    Handles what the single-walk version did not: a LIST of walks (``core._walks``
    names them ``rw``, ``rw2``, ...), the ``-1`` sentinel for "no walk term on
    this day", and the ``<series>|`` prefix observation walks carry. Dropping any
    of those silently forecast a different model from the one that was fitted.
    """
    terms = [rw] if not isinstance(rw, (list, tuple)) else list(rw)
    total = 0.0
    for i, term in enumerate(terms):
        tag = f"{prefix}rw" if i == 0 else f"{prefix}rw{i + 1}"
        walk = take(tag, required=False)
        if walk is None:
            raise ValueError(
                f"forecasting needs the walk {tag!r} in the posterior; have it "
                "recorded by build_epidemia_model, or drop the rw term"
            )
        walk = np.asarray(walk)
        if walk.ndim == 2:
            walk = walk[:, None, :]
        index = np.asarray(term.index, dtype=int)
        if index.shape != (M, T_fit):
            raise ValueError(
                f"{tag}.index must be {(M, T_fit)}, got {index.shape}"
            )
        proc = np.arange(M) if term.by_region else np.zeros(M, dtype=int)
        if term.by_region and walk.shape[1] != M:
            raise ValueError(
                f"{tag}.by_region=True needs {M} walks, posterior has "
                f"{walk.shape[1]}"
            )
        if rw_forecast == "hold":
            held = np.minimum(np.arange(T_ext)[None, :], (lengths - 1)[:, None])
            idx_ext = index[np.arange(M)[:, None], held]
        elif rw_forecast == "draw":
            idx_ext = _extend_rw_index(index, lengths, T_ext)
            need = int(idx_ext.max()) + 1
            scale = take(f"{tag}_scale", required=False)
            if need > walk.shape[2] and scale is None:
                raise ValueError(
                    f"rw_forecast='draw' needs {tag}_scale in the posterior to "
                    "draw increments past the fitted window. Pass "
                    "rw_forecast='hold' to freeze the walk instead."
                )
            if scale is not None:
                walk = _extend_walk(walk, need, scale, rng)
        else:
            raise ValueError(
                f"rw_forecast must be 'draw' or 'hold', got {rw_forecast!r}"
            )
        if idx_ext.max() >= walk.shape[2]:
            raise ValueError(
                f"{tag} index reaches step {idx_ext.max()} but the posterior "
                f"walk has {walk.shape[2]} step(s)"
            )
        # A pinned zero level so index -1 contributes nothing, exactly as
        # core._random_walk does when it builds the model.
        pad = np.concatenate(
            [np.zeros((walk.shape[0], walk.shape[1], 1)), walk], axis=2)
        total = total + pad[:, proc[:, None], idx_ext + 1]
    return total


def _draw_series(E, obs: ObsModel, aux, rng):
    """Draw from one series' family, matching how ``core`` parameterises it.

    ``core`` and :func:`epidemia.predict.posterior_predict` do not use the same
    auxiliary parameterisation for two families, so translate rather than pass
    ``aux`` through blindly:

    * ``quasi_poisson``: ``core`` fits ``alpha = mu / aux`` (R's
      ``neg_binomial_2(E, E / aux)``), so ``aux`` is the EXCESS of the
      variance-to-mean ratio over one; ``posterior_predict`` takes ``mu / aux``
      and so needs it unchanged.
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
        return posterior_predict(E, "quasi_poisson", aux=a, rng=rng)
    if fam == "log_normal":
        return posterior_predict(np.log(E) + a**2 / 2.0, "log_normal",
                                 aux=a, rng=rng)
    return posterior_predict(E, fam, aux=a, rng=rng)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def forecast(idata, panel: PanelData, obs_models, config: EpiModelConfig,
             newdata=None, draws=None, seed=None, series=None,
             group="country", date="date", rw_forecast="draw"):
    """Forecast every latent and observed series from a fitted model.

    Rebuilds the model's design over a (possibly longer) window, reconstructs
    the linear predictor from the posterior draws exactly as
    :func:`epidemia.core.build_epidemia_model` does::

        eta        = b0[m] + sum_k (beta_k + b[m, k]) X[m, t, k] + rw[p(m), tau(m, t)]
        Rt_unadj   = config.R_link_K * sigmoid(eta)

    then runs the renewal recursion (with the susceptibility adjustment when
    ``config.pop_adjust``) and, per observation series::

        rate = link_K * sigmoid(obs eta)
        E    = rate * conv(infections, i2o) + 1e-15

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
    ``rw_forecast`` controls what the random walk does past the fitted window.
    ``"draw"`` (the default, and what R does) continues it by drawing new
    increments at the walk's own fitted scale, so the forecast fans out.
    ``"hold"`` freezes each region's walk at its last fitted step, giving the
    narrower "R_t stays where it ended" forecast this function used to produce
    unconditionally.
    """
    # A latent fit CAN be forecast: R rebuilds infections_raw over the new
    # window, marks the unseen periods, and continues the latent process with
    # normal_lb_rng(mu, sigma, 0) (gen_infections_pp.stan:36-44). We do the
    # same below -- the fitted window is read back from the posterior and the
    # horizon is drawn, rather than run through the deterministic recursion,
    # which would be a different model from the one that was fitted.
    latent_fit = bool(getattr(config, "latent", False))

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

    # build_epidemia_model subtracts the FITTED design's column means when
    # center=True (core.py), so the coefficients are on the centred scale. A
    # forecast that skips this evaluates them against uncentred covariates and
    # shifts the whole linear predictor by xbar . beta.
    _xbar = (X_fit.reshape(-1, K).mean(axis=0)
             if (getattr(config, "center", False) and K) else None)

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

    if _xbar is not None:
        X_ext = X_ext - _xbar[None, None, :]

    # ---- posterior draws -------------------------------------------------
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    # b0 is the per-region intercept deviation. A model built with
    # region_effects=False never creates it -- a single-population fit has no
    # regions to vary over -- so it must not be required here. Rt_unadj is
    # always present and gives the draw count either way.
    b0_all = _stack_draws(post, "b0", required=False)
    n_samples = (b0_all.shape[0] if b0_all is not None
                 else _stack_draws(post, "Rt_unadj").shape[0])
    if draws is not None and int(draws) < n_samples:
        idx = np.sort(rng.choice(n_samples, size=int(draws), replace=False))
    else:
        idx = np.arange(n_samples)
    S = idx.shape[0]

    def take(name, required=True):
        arr = _stack_draws(post, name, required=required)
        return None if arr is None else np.asarray(arr, dtype=float)[idx]

    eta = np.zeros((S, M, T_ext))
    if config.intercept:
        eta += take("intercept").reshape(S, 1, 1)
    if b0_all is not None:
        eta += np.asarray(b0_all, dtype=float)[idx][:, :, None]  # (S, M)

    if K:
        coef = take("beta").reshape(S, 1, K)
        # region_effects=False builds no per-region slope deviations, so `b` is
        # absent -- a fully pooled fit with covariates could not be forecast at
        # all while this was required.
        b = take("b", required=False)
        if b is not None:
            coef = coef + np.asarray(b, dtype=float).reshape(S, M, K)
        eta += np.einsum("smk,mtk->smt",
                         np.broadcast_to(coef, (S, M, K)), X_ext)

    if config.rw is not None:
        eta += _walk_eta(config.rw, "", take, rng, M, T_fit, T_ext, lengths,
                         rw_forecast)

    Rt_unadj = _apply_link(getattr(config, "link", "scaled_logit"), eta,
                           float(config.R_link_K))

    # ---- infections ------------------------------------------------------
    seed_draws = take("seed")                                    # (S, M)
    # With epiinf(latent = TRUE) the post-seeding infections are parameters, so
    # the recursion supplies their MEAN and the value is drawn around it. Over
    # the fitted window the draws already exist, so they are read back rather
    # than re-simulated -- R marks those periods and skips them the same way.
    latent_sd = observed_inf = None
    if latent_fit:
        aux = take("inf_aux", required=False)
        if aux is None:
            raise ValueError(
                "forecasting a latent=True fit needs 'inf_aux' (the latent "
                "dispersion) in the posterior, and this fit does not carry it."
            )
        a = np.repeat(np.asarray(aux, dtype=float).reshape(S, 1), M,
                      axis=1).reshape(S * M, 1)
        if getattr(config, "fixed_vtm", True):
            def latent_sd(mean, _a=a):
                return np.sqrt(_a * mean)
        else:
            def latent_sd(mean, _a=a):
                return _a * mean
        fitted_inf = take("infections", required=False)
        if fitted_inf is not None:
            observed_inf = np.asarray(fitted_inf, dtype=float)

    if config.pop_adjust:
        if panel.pops is None:
            raise ValueError("config.pop_adjust=True requires PanelData.pops")
        pops = np.asarray(panel.pops, dtype=float)
        s0 = take("S0", required=False)
        susc0 = pops[None, :] if s0 is None else s0 * pops[None, :]
        # Removal (vaccination) is part of the fitted recursion, so a forecast
        # that ignored it would not reproduce the fit in-sample. Beyond the
        # fitted window the last known rate carries forward, like the covariates.
        rm = None
        if getattr(config, "rm", None) is not None:
            rm_fit = np.asarray(config.rm, dtype=float)
            T_ext = Rt_unadj.shape[-1]
            rm = np.empty((rm_fit.shape[0], T_ext))
            n = min(rm_fit.shape[1], T_ext)
            rm[:, :n] = rm_fit[:, :n]
            if T_ext > n:
                rm[:, n:] = rm_fit[:, [-1]]
        infections, susceptible = _renewal(
            Rt_unadj, config.gen, seed_draws, config.seed_days,
            pops=pops[None, :], susc0=susc0, rm=rm,
            latent_sd=latent_sd, rng=rng, observed=observed_inf,
        )
        Rt = Rt_unadj * susceptible / pops[None, :, None]
    else:
        infections, susceptible = _renewal(
            Rt_unadj, config.gen, seed_draws, config.seed_days,
            latent_sd=latent_sd, rng=rng, observed=observed_inf,
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
        if getattr(o, "rw", None) is not None:
            # A series' own ascertainment walk. Dropping it forecast a model
            # with a FLAT ascertainment rate, which is not the one that was
            # fitted whenever epiobs carried an rw() term.
            oeta += _walk_eta(o.rw, f"{o.name}|", take, rng, M, T_fit, T_ext,
                              lengths, rw_forecast)

        rate = _apply_link(getattr(o, "link", "scaled_logit"), oeta,
                           float(o.link_K))
        # 1e-15 floor, exactly as build_epidemia_model records E_<series>
        # and as R adds inside neg_binomial_2_lpmf.
        E = expected_observations(infections, np.asarray(o.i2o, dtype=float),
                                  rate) + 1e-15
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
