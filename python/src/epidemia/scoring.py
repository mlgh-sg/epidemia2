"""Scoring probabilistic forecasts (pure NumPy, mirrors R's ``forecasting.R``).

The R package evaluates a fit with ``evaluate_forecast()`` / ``posterior_metrics()``
/ ``posterior_coverage()``: per-observation error metrics (CRPS, mean and median
absolute error) plus the empirical coverage of central credible intervals. The
functions here are the same calculations, decoupled from any model object -- they
take the observed vector and a matrix of predictive draws, so they score anything
you can sample from (a PyMC posterior predictive, a forward simulation, an
external forecast pulled from a CSV).

**Draw layout.** ``draws`` is ``(n_observations, n_draws)``: one *row* per
observation holding that observation's own predictive sample. This is the
load-bearing detail. Each observation must be scored against its own predictive
distribution; pooling every date and group into one empirical distribution and
scoring everything against that marginal is not a forecast score at all (it is
the bug that was just fixed in R -- a point-mass predictive scored 4.44 where the
answer is 0, and every early day with an observed count of zero got the same
non-zero value). Most posterior-predictive arrays come out ``(draw, time)``, so
they need a transpose on the way in.

**Placeholders.** Following R's ``epiobs_``, an observation that is ``NaN`` or
coded ``-1`` is a forecast-horizon placeholder, not a count of minus one.
:func:`evaluate_forecast` drops those rows -- and the matching predictive rows,
so the two stay aligned -- rather than scoring a forecast against a truth of -1,
which inflates the error and collapses coverage to zero.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Error metrics understood by :func:`evaluate_forecast` (R's ``ok_metrics``).
METRICS = ("crps", "mean_abs_error", "median_abs_error")

#: Observations coded with this value are forecast-horizon placeholders, not counts.
PLACEHOLDER = -1


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def _as_y_draws(y, draws):
    """Coerce to ``y`` of shape ``(N,)`` and ``draws`` of shape ``(N, S)``."""
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    draws = np.asarray(draws, dtype=float)
    if draws.ndim == 1:
        # A single observation's sample, or one draw per observation -- both are
        # meaningful, and which one it is follows from y.
        draws = draws.reshape(1, -1) if y.size == 1 else draws.reshape(-1, 1)
    if draws.ndim != 2:
        raise ValueError(f"draws must be 2-d (observation, draw); got shape {draws.shape}")
    if draws.shape[0] != y.size:
        extra = ""
        if draws.shape[1] == y.size:
            # posterior_predict-style output is (draw, observation); the whole
            # point of this module is that each row is one observation's sample.
            extra = (" -- draws looks transposed (draw, observation); pass draws.T"
                     " so each row is one observation's predictive sample")
        raise ValueError(
            f"draws must have one row per observation: got {draws.shape} for "
            f"{y.size} observation(s){extra}"
        )
    return y, draws


def _check_levels(levels) -> np.ndarray:
    """Validate credible-interval levels (percent), sorted ascending as in R."""
    lv = np.atleast_1d(np.asarray(list(levels) if levels is not None else [], dtype=float))
    if lv.size == 0:
        warnings.warn(
            "no levels provided, will use default credible intervals (50% and 95%)",
            stacklevel=3,
        )
        return np.array([50.0, 95.0])
    if np.any((lv < 0) | (lv > 100)):
        raise ValueError("all levels must be between 0 and 100 (inclusive)")
    return np.sort(lv)


def _check_metrics(metrics) -> tuple[str, ...]:
    if metrics is None:
        return METRICS
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise ValueError(
            f"unrecognised metrics {bad}; allowed metrics are {', '.join(METRICS)}"
        )
    return metrics


def _labels(n: int, group, date):
    """Group/date columns for the output tables, defaulted when not supplied.

    The R tables always carry ``group`` and ``date`` because a fit always has
    them. Here they are optional, so a single unnamed series still gets stable
    columns: one group ``"all"`` and a 0-based day index.
    """
    if group is None:
        g = np.full(n, "all", dtype=object)
    else:
        g = np.asarray(group).ravel()
        if g.size != n:
            raise ValueError(f"group has length {g.size}, expected {n}")
    if date is None:
        d = np.arange(n)
    else:
        d = np.asarray(date).ravel()
        if d.size != n:
            raise ValueError(f"date has length {d.size}, expected {n}")
    return g, d


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def crps_sample(y: float, x) -> float:
    """CRPS of one observation against one predictive sample (sorted estimator).

    Parameters
    ----------
    y : float
        The observed value.
    x : array (S,)
        Predictive draws for that observation.
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    a = (np.arange(n) + 0.5) / n
    return float(2.0 / n * np.sum(((y < x).astype(float) - a) * (x - y)))


def crps(y, draws) -> np.ndarray:
    """Continuous ranked probability score, one value per observation.

    Uses the standard sorted-sample estimator of
    :math:`\\int (F(x) - 1\\{x \\ge y\\})^2 dx`, i.e. the empirical CDF of that
    observation's draws integrated against the step function at the truth. It
    reduces to the absolute error when the predictive is a point mass, and is
    minimised (over ``y``) at the predictive median.

    Parameters
    ----------
    y : array (N,)
        Observed values.
    draws : array (N, S)
        Predictive draws, **one row per observation**. Row ``i`` alone determines
        entry ``i`` of the result -- see the module docstring for why that matters.

    Returns
    -------
    array (N,)
        CRPS of each observation, in the same units as ``y`` (smaller is better).
    """
    y, draws = _as_y_draws(y, draws)
    xs = np.sort(draws, axis=1)
    n = xs.shape[1]
    a = (np.arange(n) + 0.5) / n  # plotting positions of the sample ECDF
    ind = (y[:, None] < xs).astype(float)
    return (2.0 / n) * np.sum((ind - a) * (xs - y[:, None]), axis=1)


def absolute_error(y, draws) -> dict[str, np.ndarray]:
    """Mean and median absolute error against each observation's own draws.

    Both are taken *over the draws* of one observation (R's ``rowMeans``/row
    ``median`` of ``|draws - y|``), not over observations -- so like
    :func:`crps` they are per-observation scores.

    Returns
    -------
    dict
        ``{"mean_abs_error": (N,), "median_abs_error": (N,)}``.
    """
    y, draws = _as_y_draws(y, draws)
    dev = np.abs(draws - y[:, None])
    return {
        "mean_abs_error": dev.mean(axis=1),
        "median_abs_error": np.median(dev, axis=1),
    }


def coverage(y, draws, levels: Sequence[float] = (50, 95), group=None,
             date=None) -> pd.DataFrame:
    """Whether each observation falls inside its central credible intervals.

    Parameters
    ----------
    y : array (N,)
        Observed values.
    draws : array (N, S)
        Predictive draws, one row per observation.
    levels : sequence of float, default ``(50, 95)``
        Credible-interval levels in percent.
    group, date : array (N,), optional
        Labels carried through to the table; defaulted if omitted.

    Returns
    -------
    DataFrame
        Long form, one row per ``(observation, level)``, with columns
        ``group, date, level, tag, lower, upper, in_ci``.

    Notes
    -----
    Placeholder rows (``NaN`` / ``-1``) are *not* dropped here; call
    :func:`drop_placeholders` first, or use :func:`evaluate_forecast`, which does
    it for you.
    """
    y, draws = _as_y_draws(y, draws)
    lv = _check_levels(levels)
    g, d = _labels(y.size, group, date)

    frames = []
    for level in lv:
        lo = np.percentile(draws, (100 - level) / 2, axis=1)
        hi = np.percentile(draws, 100 - (100 - level) / 2, axis=1)
        frames.append(pd.DataFrame({
            "group": g,
            "date": d,
            "level": level,
            "tag": f"{level:g}% CI",
            "lower": lo,
            "upper": hi,
            "in_ci": (lo <= y) & (y <= hi),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Placeholders
# --------------------------------------------------------------------------


def drop_placeholders(y, draws, group=None, date=None):
    """Drop rows whose observation is ``NaN`` or the ``-1`` forecast placeholder.

    R's ``epiobs_`` documents outcomes as "positive, NA, or coded -1 (for
    forecasting)", and the multiple-observations tutorial builds ``newdata`` that
    way. Scoring such a row treats the truth as -1. The predictive rows (and any
    labels) are subset alongside so everything stays aligned.

    The test is R's ``!is.na(y) & y >= 0``, so *any* negative outcome counts as a
    placeholder -- modelled outcomes are counts, and the only negative one the
    package ever produces is the ``-1`` code.

    Returns
    -------
    tuple
        ``(y, draws, group, date)`` restricted to the scorable rows; ``group`` and
        ``date`` are ``None`` if they were not supplied.
    """
    y, draws = _as_y_draws(y, draws)
    keep = np.isfinite(y) & (y >= 0)
    if group is not None:
        group = np.asarray(group).ravel()[keep]
    if date is not None:
        date = np.asarray(date).ravel()[keep]
    return y[keep], draws[keep], group, date


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class ForecastEvaluation:
    """Result of :func:`evaluate_forecast`: R's named list of two dataframes.

    Attributes
    ----------
    error : DataFrame
        One row per observation: ``group, date`` plus the requested metrics.
    coverage : DataFrame
        One row per ``(observation, level)``: ``group, date, level, tag, lower,
        upper, in_ci``.
    """

    error: pd.DataFrame
    coverage: pd.DataFrame

    # R returns list(error=, coverage=), so accept ["error"] as well as .error --
    # code ported straight across keeps working either way.
    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("error", "coverage")


def evaluate_forecast(
    y,
    draws,
    group=None,
    date=None,
    levels: Sequence[float] = (50, 95),
    metrics: Iterable[str] | str | None = None,
) -> ForecastEvaluation:
    """Score a probabilistic forecast: error metrics and interval coverage.

    Mirrors R's ``evaluate_forecast()``, minus the model plumbing (the caller
    supplies the predictive draws directly).

    Parameters
    ----------
    y : array (N,)
        Observed counts. ``NaN`` and negative entries (the ``-1`` forecast code)
        are placeholders and are dropped, along with their predictive rows --
        see :func:`drop_placeholders`.
    draws : array (N, S)
        Predictive draws, one row per observation.
    group, date : array (N,), optional
        Labels for the output tables; defaulted to ``"all"`` and a day index.
    levels : sequence of float, default ``(50, 95)``
        Credible-interval levels in percent, for the coverage table.
    metrics : str or sequence of str, optional
        Subset of :data:`METRICS`; all of them by default.

    Returns
    -------
    ForecastEvaluation
        With ``.error`` and ``.coverage`` dataframes (also reachable as
        ``result["error"]`` / ``result["coverage"]``, like R's named list).
    """
    metrics = _check_metrics(metrics)
    lv = _check_levels(levels)
    y, draws = _as_y_draws(y, draws)
    g, d = _labels(y.size, group, date)
    y, draws, g, d = drop_placeholders(y, draws, g, d)

    error = pd.DataFrame({"group": g, "date": d})
    if "crps" in metrics:
        error["crps"] = crps(y, draws)
    if {"mean_abs_error", "median_abs_error"} & set(metrics):
        ae = absolute_error(y, draws)
        for m in ("mean_abs_error", "median_abs_error"):
            if m in metrics:
                error[m] = ae[m]

    return ForecastEvaluation(
        error=error,
        coverage=coverage(y, draws, levels=lv, group=g, date=d),
    )


def posterior_metrics(y, draws, group=None, date=None, metrics=None) -> pd.DataFrame:
    """The ``error`` table of :func:`evaluate_forecast` (R's ``posterior_metrics``)."""
    return evaluate_forecast(y, draws, group=group, date=date, metrics=metrics).error


def posterior_coverage(y, draws, group=None, date=None,
                       levels: Sequence[float] = (50, 95)) -> pd.DataFrame:
    """The ``coverage`` table of :func:`evaluate_forecast` (R's ``posterior_coverage``)."""
    return evaluate_forecast(y, draws, group=group, date=date, levels=levels).coverage
