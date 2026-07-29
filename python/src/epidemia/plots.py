"""Plotting with plotnine (a grammar of graphics for Python).

These helpers mirror the R package's plots — credible-interval ribbons in a
single-hue sequential palette (wider interval = lighter) with the posterior
median on top — and ship a clean, publication-oriented theme.

**Multi-region models.** As in R, :func:`plot_rt`, :func:`plot_infections` and
:func:`plot_obs` handle both the single-population model and the multi-region
(:mod:`epidemia.multilevel`) one. Pass the :class:`~epidemia.multilevel.MultilevelData`
you fitted as ``data`` and every region is drawn in its own panel, on real dates,
with each region's padded days dropped. ``group="Italy"`` restricts to one region.

**Saving.** Every plotting function writes a PNG by default (``save=True``) into
:func:`figure_dir` — ``$EPIDEMIA_FIGDIR`` if set, else ``./figures``. Pass
``save=False`` to skip, or ``save="name"``/``save="path/to/f.png"`` to choose the
file. The returned object is still a ``plotnine.ggplot``, so it also renders
inline in a notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_pointrange,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
#
# Chosen for print and for colour-vision deficiency rather than for screen
# punch: a single-hue sequential ramp per quantity, so a wider interval is
# always lighter, and the four *categorical* roles (observed, fitted,
# counterfactual, forecast) are separated in both hue and lightness. These are
# steps from ColorBrewer's Blues/Greens/Purples/Oranges, which are the ramps
# bayesplot and the R vignettes use.

#: Posterior interval fills, darkest (narrowest) to lightest (widest).
# Steps are far enough apart in LIGHTNESS that a narrow band reads as distinct
# from a wide one even at small facet sizes and in greyscale. The previous ramp
# put 30 and 50 on the SAME hex, so a 50/95 plot showed one visible band.
# Mid-range steps rather than the extremes of each ramp: the darkest band still
# has to sit UNDER a black median line, and a near-black innermost band swallows
# it. These are the ColorBrewer steps R's vignettes use.
_GREENS = {30: "#238b45", 50: "#238b45", 60: "#74c476", 90: "#c7e9c0", 95: "#e5f5e0"}
_BLUES = {30: "#2171b5", 50: "#2171b5", 60: "#6baed6", 90: "#c6dbef", 95: "#deebf7"}
_PURPLES = {30: "#6a51a3", 50: "#6a51a3", 60: "#9e9ac8", 90: "#dadaeb", 95: "#efedf5"}
_ORANGES = {30: "#d94801", 50: "#d94801", 60: "#fd8d3c", 90: "#fdd0a2", 95: "#fee6ce"}

#: Categorical roles. Distinguishable under deuteranopia and in greyscale.
COLORS = {
    "rt": "#238b45",            # green: reproduction numbers
    "infections": "#2171b5",    # blue: latent infections
    "observed": "#b2182b",      # red: the data
    "in_sample": "#b2182b",
    "out_of_sample": "#2166ac",
    "counterfactual": "#6a51a3",  # purple: "what if" -- never the same hue as the fit
    "forecast": "#d94801",      # orange: projection beyond the fit
    "median": "#252525",
}


def theme_epidemia(base_size: float = 11, rotate_x: float = 45):
    """A clean, publication-oriented plotnine theme.

    ``rotate_x`` tilts the x tick labels. Dates on a faceted panel are long
    enough that horizontal labels collide -- in an 11-country facet they overlap
    into an unreadable smear -- so the default is 45 degrees, as the Financial
    Times and Our World in Data house styles do.
    """
    return theme_minimal(base_size=base_size) + theme(
        figure_size=(8, 4.5),
        dpi=140,
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(color="#ededed", size=0.35),
        panel_spacing=0.035,
        axis_title=element_text(size=base_size),
        axis_text=element_text(size=base_size - 2, color="#4d4d4d"),
        axis_text_x=element_text(size=base_size - 2, color="#4d4d4d",
                                 angle=rotate_x,
                                 ha="right" if rotate_x else "center"),
        strip_text=element_text(size=base_size - 1, weight="bold",
                                color="#262626"),
        legend_position="top",
        legend_title=element_text(size=base_size - 1),
        legend_key_size=12,
        plot_title=element_text(size=base_size + 1, weight="bold"),
    )


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def figure_dir() -> Path:
    """Directory that :func:`save_plot` writes into (created if absent).

    ``$EPIDEMIA_FIGDIR`` if set, otherwise ``./figures``.
    """
    d = Path(os.environ.get("EPIDEMIA_FIGDIR", "figures"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_plot(p, name, width=None, height=None, dpi=140, verbose=False):
    """Write ``p`` to a PNG and return the :class:`~pathlib.Path` written.

    ``name`` may be a bare stem (``"rt"`` -> ``<figure_dir>/rt.png``), a name with
    a suffix (``"rt.pdf"``), or a full path (anything containing a separator).
    """
    path = Path(name)
    if path.suffix == "":
        path = path.with_suffix(".png")
    if len(path.parts) == 1:
        path = figure_dir() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if width is not None:
        kw["width"] = width
    if height is not None:
        kw["height"] = height
    p.save(path, dpi=dpi, verbose=verbose, **kw)
    print(f"[epidemia] saved {path}")
    return path


_FACET_NCOL = 4


def _panel_size(n_panels):
    """Canvas size for a plot with ``n_panels`` facets (the theme's default at 1).

    Applied to the ggplot itself, not just at save time, so an 11-country facet
    is legible inline in a notebook as well as on disk.
    """
    if n_panels <= 1:
        return (8.0, 4.5)
    ncol = min(_FACET_NCOL, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    return (3.1 * ncol, 2.3 * nrow + 0.8)


def _size_for_panels(p, n_panels):
    if n_panels > 1:
        p = p + theme(figure_size=_panel_size(n_panels))
    return p


def _maybe_save(p, save, default_name, n_panels=1):
    """Honour the ``save=`` argument of a plotting function."""
    if save is False or save is None:
        return p
    name = default_name if save is True else save
    width, height = _panel_size(n_panels)
    save_plot(p, name, width=width, height=height)
    return p


# --------------------------------------------------------------------------
# Posterior extraction
# --------------------------------------------------------------------------


def _draws(idata, var):
    """Posterior draws of ``var`` with chain/draw merged, other dims preserved.

    Returns ``(array, dims)`` where ``array`` has shape ``(draws, *rest)`` and
    ``dims`` are the names of ``rest``. Unlike a blind ``reshape(-1, T)`` this
    never folds a ``region`` axis into the draw axis.
    """
    da = idata.posterior[var]
    arr = np.asarray(da)  # (chain, draw, *rest)
    return arr.reshape(-1, *arr.shape[2:]), tuple(da.dims[2:])


def _is_forecast(obj):
    """True for a :class:`~epidemia.forecast.Forecast` (duck-typed, no import)."""
    return all(hasattr(obj, a) for a in
               ("regions", "dates", "lengths", "Rt", "predicted", "expected"))

def _is_multiregion(idata, var):
    # A Forecast is always per-region: it carries a `regions` list rather than
    # a posterior whose dims can be inspected.
    if _is_forecast(idata):
        return True
    return "region" in idata.posterior[var].dims


def available_series(idata):
    """Names of the observation series present in ``idata``.

    ``build_epidemia_model`` records one ``E_<name>`` per series and the names
    are whatever the user called them, so they cannot be guessed.
    """
    return sorted(
        v[2:] for v in idata.posterior.data_vars if str(v).startswith("E_")
    )


def _families(idata):
    """Observation families recorded on the fit by ``fit_epidemia``."""
    import json
    raw = getattr(idata, "attrs", {}).get("epidemia_families")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}


def _predictive_draws(idata, var, series, obs_model=None, seed=0):
    """Posterior *predictive* draws for one series, or None if unavailable.

    R's ``plot_obs`` bands ``posterior_predict()``, so its ribbons carry the
    observation family's noise on top of parameter uncertainty. Banding the mean
    instead -- which these plots used to do -- gives a credible interval on the
    expected count, several times too narrow to compare against the data it is
    drawn over.

    Needs the family (from ``obs_model`` or recorded on the fit) and, for
    families that have one, the ``<series>|aux`` draws.
    """
    from .predict import posterior_predict

    family = getattr(obs_model, "family", None) or _families(idata).get(series)
    if family is None:
        return None
    arr, dims = _draws(idata, var)                    # (S, ...) expected values
    aux = None
    for cand in (f"{series}|aux", "aux", f"{series}|reciprocal_dispersion"):
        if cand in idata.posterior:
            aux = _draws(idata, cand)[0]
            break
    if aux is not None:
        aux = np.asarray(aux).reshape(aux.shape[0], *([1] * (arr.ndim - 1)))
    rng = np.random.default_rng(seed)
    try:
        return posterior_predict(np.asarray(arr), family, aux=aux, rng=rng)
    except Exception:
        return None


def _pick_obs_var(idata, series=None):
    """The expected-observation variable for ``series``.

    ``series`` names one of :func:`available_series`. Omitting it is only
    unambiguous when the model has a single series: the previous behaviour --
    scanning a fixed ("E_obs", "E_deaths") tuple and taking the first hit --
    made every other name unreachable and silently picked one of several.
    """
    if series is not None:
        var = f"E_{series}"
        if var not in idata.posterior:
            raise KeyError(
                f"no series named {series!r} in the posterior; "
                f"available: {available_series(idata) or 'none'}"
            )
        return var

    found = available_series(idata)
    if len(found) == 1:
        return f"E_{found[0]}"
    if not found:
        raise KeyError(
            "no expected-observation variable found in the posterior "
            f"(have {list(idata.posterior.data_vars)})"
        )
    raise ValueError(
        f"this fit has {len(found)} observation series ({', '.join(found)}); "
        "pass series= to say which one to plot"
    )


def _interval_frame(draws, x, levels):
    """Long dataframe of central credible intervals at each ``level`` (percent).

    The ``level`` categorical is ordered **widest-first**, which is load-bearing:
    geom_ribbon draws one poly per group in category-code order and they all share
    a zorder, so the last group drawn wins. Ordering it narrowest-first paints the
    widest band over every narrower one -- leaving only the outermost band
    visible while the legend still advertises the rest.
    """
    rows = []
    for lv in sorted(levels, reverse=True):
        lo = np.percentile(draws, (100 - lv) / 2, axis=0)
        hi = np.percentile(draws, 100 - (100 - lv) / 2, axis=0)
        rows.append(pd.DataFrame({"x": x, "lower": lo, "upper": hi, "level": str(lv)}))
    df = pd.concat(rows, ignore_index=True)
    df["level"] = pd.Categorical(
        df["level"], categories=[str(lv) for lv in sorted(levels, reverse=True)]
    )
    return df


def _median_frame(draws, x):
    return pd.DataFrame({"x": x, "median": np.median(draws, axis=0)})



def _drop_incomplete(band, med):
    """Drop rows the rolling mean left undefined (its incomplete windows)."""
    if band is not None and len(band):
        band = band[np.isfinite(band["lower"]) & np.isfinite(band["upper"])]
    if med is not None and len(med):
        med = med[np.isfinite(med["median"])]
    return band, med


def _as_groups(group, groups):
    """Reconcile R's ``groups`` vector with the older singular ``group``."""
    if groups is None:
        return None if group is None else (
            [str(g) for g in group] if isinstance(group, (list, tuple, set))
            else [str(group)])
    if isinstance(groups, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(g) for g in groups]
    return [str(groups)]


def _region_frame(idata, var, data, levels, group=None, transform=None,
                  draws=None):
    """Per-region interval + median frames on real dates, padded days dropped.

    ``data`` is the :class:`~epidemia.multilevel.MultilevelData` that was fitted;
    it supplies each region's genuine dates and length, without which column ``t``
    would be a *different date* in every region.
    """
    arr, dims = _draws(idata, var)  # (draws, region, time)
    if draws is not None:
        arr = np.asarray(draws)     # predictive draws, same shape
    if dims[:1] != ("region",):
        raise ValueError(f"expected a region dim on {var!r}, got dims {dims}")
    regions = [str(r) for r in idata.posterior.coords["region"].values]
    want = _as_groups(group, None)
    keep = regions if want is None else want
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    bands, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(data.lengths[data.regions.index(r)])
        x = pd.to_datetime(data.dates[data.regions.index(r)])[:n]
        d = arr[:, m, :n]
        if transform is not None:
            d = transform(d, data.regions.index(r))
        b = _interval_frame(d, x, levels)
        b["region"] = r
        bands.append(b)
        md = _median_frame(d, x)
        md["region"] = r
        meds.append(md)
    return pd.concat(bands, ignore_index=True), pd.concat(meds, ignore_index=True), keep


def _observed_frame(data, group=None, obs_model=None, transform=None):
    """Long frame of the genuinely **observed** counts per region on real dates.

    Days ``prepare_panel`` masked out (a missing count) are dropped rather than
    drawn. ``data.deaths`` is zero-filled on those days -- only ``mask`` records
    that the zero is a placeholder -- so reading ``deaths`` alone would plot a
    fabricated zero-death observation the model was never fit to.
    """
    # MultilevelData carries the counts on the panel itself (`deaths`/`mask`);
    # PanelData does not -- with several series the observations live on each
    # ObsModel, so they have to be supplied.
    if obs_model is not None:
        counts, valid = np.asarray(obs_model.y), np.asarray(obs_model.mask)
    elif hasattr(data, "deaths"):
        counts, valid = np.asarray(data.deaths), np.asarray(data.mask)
    else:
        raise ValueError(
            "this panel carries no observed counts (PanelData holds only the "
            "design); pass obs_model=<the ObsModel for this series> so the "
            "observations can be overlaid"
        )

    want = _as_groups(group, None)
    keep = data.regions if want is None else want
    unknown = set(keep) - set(data.regions)
    if unknown:
        raise ValueError(
            f"unknown region(s) {sorted(unknown)}; have {list(data.regions)}")
    rows = []
    for r in keep:
        m = data.regions.index(r)
        n = int(data.lengths[m])
        obs = np.asarray(counts[m, :n], dtype=float)
        obs[~valid[m, :n]] = np.nan                  # missing is not a zero
        if transform is not None:
            # R applies cumulative/by_100k to the DATA as well as the draws
            # (R/plots_epi.R:211-225). Transforming only the bands leaves the
            # bars on a different scale from the ribbon drawn over them.
            obs = transform(obs[None, :], data.regions.index(r))[0]
        df = pd.DataFrame({
            "x": pd.to_datetime(data.dates[m])[:n], "obs": obs, "region": r,
        })
        rows.append(df[np.isfinite(df["obs"])])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Core ribbon plot
# --------------------------------------------------------------------------


def _level_colours(palette, levels):
    """Map each level to a colour, interpolating for levels the palette lacks.

    ``levels`` is a documented public parameter, so an unlisted value (e.g.
    ``levels=(80,)``, or the ``seq(10, 90, 10)`` the R vignette uses for prior
    checks) must not blow up on a dictionary lookup. Ramp light->dark with the
    interval width, matching the fixed palette's intent: wider = lighter.
    """
    known = sorted(palette)
    out = {}
    for lv in levels:
        if lv in palette:
            out[str(lv)] = palette[lv]
            continue
        nearest = min(known, key=lambda k: abs(k - lv))
        out[str(lv)] = palette[nearest]
    return out


def _ribbon_plot(band, med, palette, levels, ylab, xlab, hline=None, facet=0,
                 title=None, obs=None, obs_kind="point", step=False):
    """``facet`` is the number of region panels (0/1 => no faceting)."""
    cols = _level_colours(palette, levels)
    p = (
        ggplot()
        + geom_ribbon(band, aes("x", ymin="lower", ymax="upper", fill="level"))
        # Bands are drawn widest-first (see _interval_frame); the legend still
        # reads narrowest-first, which is the order people expect to see.
        + scale_fill_manual(values=cols, name="Credible interval (%)",
                            breaks=[str(lv) for lv in sorted(levels)])
    )
    if obs is not None:
        has_period = "period" in getattr(obs, "columns", [])
        # Points, not filled columns. A geom_col at daily width over a 3-month
        # window merges into a solid block that hides the ribbon underneath --
        # which is exactly what made the England case plot unreadable. Points
        # keep every observation visible AND let the bands show through, which
        # is how the FT and Our World in Data draw daily counts.
        if has_period:
            pal = {"In-sample": COLORS["in_sample"],
                   "Out-of-sample": COLORS["out_of_sample"]}
            if obs_kind == "col":
                p = p + geom_col(obs, aes("x", "obs", fill="period"),
                                 alpha=0.55, width=0.75)
                p = p + scale_fill_manual(
                    values={**{str(lv): c for lv, c in cols.items()}, **pal},
                    name="Credible interval (%)",
                    breaks=[str(lv) for lv in sorted(levels)])
            else:
                p = p + geom_point(obs, aes("x", "obs", color="period"),
                                   size=0.9, alpha=0.75, stroke=0)
                p = p + scale_color_manual(values=pal, name="")
        elif obs_kind == "col":
            p = p + geom_col(obs, aes("x", "obs"), fill=COLORS["observed"],
                             alpha=0.45, width=0.75)
        else:
            p = p + geom_point(obs, aes("x", "obs"), color=COLORS["observed"],
                               size=0.9, alpha=0.75, stroke=0)
    if step:
        # R's plot_rt(step = TRUE): a covariate-driven R_t is piecewise constant,
        # so a step reads more honestly than an interpolating line.
        from plotnine import geom_step

        p = p + geom_step(med, aes("x", "median"), color="black", size=0.6)
    else:
        p = p + geom_line(med, aes("x", "median"), color="black", size=0.6)
    p = p + labs(x=xlab, y=ylab)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dotted", color="#555555")
    if title:
        p = p + labs(title=title)
    p = p + theme_epidemia()
    if facet:
        p = p + facet_wrap("region", scales="free_y", ncol=_FACET_NCOL)
        p = _size_for_panels(p, facet)  # `facet` carries the panel count
    return p


def _window(band, med, obs_df, dates):
    """Restrict the frames to ``dates=(start, end)``; either end may be None."""
    if not dates:
        return band, med, obs_df
    lo, hi = dates
    def cut(df):
        if df is None or not len(df):
            return df
        keep = pd.Series(True, index=df.index)
        if lo is not None:
            keep &= df["x"] >= pd.to_datetime(lo)
        if hi is not None:
            keep &= df["x"] <= pd.to_datetime(hi)
        return df[keep]
    return cut(band), cut(med), cut(obs_df)


def _pseudo_log(x, sigma=1.0, base=10.0):
    """R's ``scales::pseudo_log_trans``: ``asinh(x / 2s) / log(base)``.

    Defined at (and near) zero, unlike a plain log. That matters here because
    the series being drawn -- infections during the seeding period, deaths early
    in an epidemic -- routinely contain zeros, and ``scale_y_log10`` drops those
    rows silently rather than showing them.
    """
    x = np.asarray(x, dtype=float)
    return np.arcsinh(x / (2.0 * sigma)) / np.log(base)


def _pseudo_log_inv(y, sigma=1.0, base=10.0):
    y = np.asarray(y, dtype=float)
    return 2.0 * sigma * np.sinh(y * np.log(base))


def _log_scale(p, log):
    """Pseudo-log the y axis, matching R's ``trans = "pseudo_log"``.

    R's base_plot sets ``trans = ifelse(log, "pseudo_log", "identity")``
    (R/plots_epi.R:1018), NOT a log10 scale. The difference is not cosmetic: a
    log10 axis is undefined at zero, so every zero-count day disappears from the
    plot without comment.
    """
    if not log:
        return p
    from mizani.transforms import trans_new
    from plotnine import scale_y_continuous

    pseudo = trans_new("pseudo_log", _pseudo_log, _pseudo_log_inv)
    return p + scale_y_continuous(trans=pseudo)



def _check_smooth(smooth, n_days):
    """Validate a smoothing window the way R's ``check_smooth`` does.

    R warns and falls back to no smoothing when the window is not a positive
    integer or is at least as long as the shortest group's series
    (R/plots_epi.R:920-935). Silently returning a series of all-NaN -- which a
    window wider than the data would do here -- is worse than not smoothing.
    """
    import warnings

    try:
        k = int(smooth)
    except (TypeError, ValueError):
        warnings.warn(f"smooth={smooth!r} is not an integer; not smoothing.",
                      stacklevel=3)
        return 1
    if k != smooth:
        warnings.warn(f"smooth={smooth!r} is not an integer; using {k}.",
                      stacklevel=3)
    if k < 1:
        warnings.warn(f"smooth={k} must be positive; not smoothing.", stacklevel=3)
        return 1
    if k >= n_days:
        warnings.warn(
            f"smooth={k} is not shorter than the {n_days}-day series; "
            "not smoothing.", stacklevel=3)
        return 1
    return k


def _draw_transform(cumulative=False, smooth=None, by_100k=False, pops=None):
    """A transform applied to a region's ``(draws, time)`` array before summarising.

    It has to act on the DRAWS, not on the quantiles: a quantile of a cumulative
    sum is not the cumulative sum of the quantiles, and the same goes for a
    rolling mean. Scaling alone would commute, but it is cheaper to keep all
    three in one place.
    """
    if not (cumulative or smooth or by_100k):
        return None

    def apply(d, region_index):
        out = d
        if cumulative:
            out = np.cumsum(out, axis=-1)
        if smooth:
            k = _check_smooth(smooth, out.shape[-1])
            if k > 1:
                # Centred rolling mean with the incomplete windows DROPPED, as
                # R's smooth_obs does (zoo::rollmean(fill = NA) then
                # complete.cases). A "same" convolution instead pads with
                # implicit zeros and keeps the edges, which biases the first and
                # last floor(k/2) days toward zero -- exactly the early-epidemic
                # days people read most closely.
                kernel = np.ones(k) / k
                rolled = np.apply_along_axis(
                    lambda v: np.convolve(v, kernel, mode="valid"), -1, out)
                pad_lo = (k - 1) // 2
                pad_hi = (k - 1) - pad_lo
                out = np.concatenate([
                    np.full(out.shape[:-1] + (pad_lo,), np.nan),
                    rolled,
                    np.full(out.shape[:-1] + (pad_hi,), np.nan),
                ], axis=-1)
        if by_100k:
            if pops is None:
                raise ValueError(
                    "by_100k=True needs region populations; pass data= whose "
                    "PanelData carries .pops"
                )
            out = out / (np.asarray(pops, dtype=float)[region_index] / 1e5)
        return out

    return apply




def _date_axis(p, date_breaks=None, date_format=None, span_days=None,
               n_panels=1):
    """R's ``date_breaks`` / ``date_format``, plus a sane automatic default.

    Left to itself plotnine puts full ISO dates at every default break, which on
    a faceted panel collide into an unreadable smear. When the caller has not
    asked for something specific, pick a break spacing from the span and a short
    label, and let the theme's 45-degree tilt do the rest.
    """
    from plotnine import scale_x_date

    if date_breaks is None and date_format is None and span_days is None:
        return p
    if date_breaks is None and span_days is not None:
        per_panel = 6 if n_panels <= 1 else 4
        weeks = max(1, int(round(span_days / 7 / per_panel)))
        date_breaks = ("1 month" if weeks >= 4 else
                       "2 weeks" if weeks >= 2 else "1 week")
    if date_format is None:
        date_format = "%d %b" if (span_days or 0) < 400 else "%b %Y"
    kw = {}
    if date_breaks is not None:
        kw["date_breaks"] = date_breaks
    kw["date_labels"] = date_format
    try:
        return p + scale_x_date(**kw)
    except Exception:      # a non-date x axis (integer days)
        return p


def _span_days(*frames):
    """Number of days the plotted frames cover, or None if x is not a date."""
    for df in frames:
        if df is None or not len(df) or "x" not in getattr(df, "columns", []):
            continue
        x = df["x"]
        if not pd.api.types.is_datetime64_any_dtype(x):
            return None
        return int((x.max() - x.min()).days) or 1
    return None




_FORECAST_VARS = {"Rt": "Rt", "Rt_unadj": "Rt_unadj", "infections": "infections"}


def _forecast_draws(fc, var, series=None, predictive=True):
    """Draws of ``var`` from a Forecast, shaped ``(draws, region, time)``."""
    if var in _FORECAST_VARS:
        return np.asarray(getattr(fc, _FORECAST_VARS[var]))
    name = series if series is not None else (
        var[2:] if str(var).startswith("E_") else var)
    source = fc.predicted if predictive else fc.expected
    if name not in source:
        raise ValueError(
            f"this forecast has no series {name!r}; have {sorted(source)}"
        )
    return np.asarray(source[name])


def _forecast_frame(fc, arr, levels, group=None, transform=None):
    """Per-region interval + median frames from a Forecast, padding dropped."""
    regions = [str(r) for r in fc.regions]
    want = _as_groups(group, None)
    keep = regions if want is None else want
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    bands, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(fc.lengths[m])
        x = pd.to_datetime(fc.dates[m])[:n]
        d = arr[:, m, :n]
        if transform is not None:
            d = transform(d, m)
        bf = _interval_frame(d, x, levels)
        bf["region"] = r
        bands.append(bf)
        mf = _median_frame(d, x)
        mf["region"] = r
        meds.append(mf)
    return (pd.concat(bands, ignore_index=True),
            pd.concat(meds, ignore_index=True), keep)


def _forecast_observed(fc, obs_model, group=None, n_fitted=None,
                       transform=None):
    """Observed counts over a forecast window, labelled in- vs out-of-sample.

    R's ``parse_new_data`` tags each plotted point "In-sample" or
    "Out-of-sample" by joining against the fitted data and maps that to the bar
    fill (R/plots_epi.R:677-721). Without it a forecast plot gives no visual cue
    about which points the model actually saw.
    """
    if obs_model is None:
        return None
    counts = np.asarray(obs_model.y, dtype=float)
    valid = np.asarray(obs_model.mask, dtype=bool)
    fitted_to = counts.shape[1] if n_fitted is None else int(n_fitted)

    regions = [str(r) for r in fc.regions]
    want = _as_groups(group, None)
    keep = regions if want is None else want
    rows = []
    for r in keep:
        m = regions.index(r)
        n = min(int(fc.lengths[m]), counts.shape[1])
        obs = counts[m, :n].astype(float)
        obs[~valid[m, :n]] = np.nan
        if transform is not None:
            obs = transform(obs[None, :], m)[0]
        x = pd.to_datetime(fc.dates[m])[:n]
        period = np.where(np.arange(n) < fitted_to, "In-sample", "Out-of-sample")
        df = pd.DataFrame({"x": x, "obs": obs, "region": r, "period": period})
        rows.append(df[np.isfinite(df["obs"])])
    return pd.concat(rows, ignore_index=True) if rows else None


def _series_plot(idata, var, data, group, levels, x, xlab, ylab, palette, hline,
                 save, default_name, title=None, obs=None, obs_kind="point",
                 dates=None, log=False, cumulative=False, smooth=None,
                 by_100k=False, step=False, obs_model=None, draws=None,
                 n_fitted=None, date_breaks=None, date_format=None):
    """Dispatch: multi-region (facet on real dates) vs single-population."""
    transform = _draw_transform(cumulative, smooth, by_100k,
                                getattr(data, "pops", None))
    if _is_forecast(idata):
        fc = idata
        arr = draws if draws is not None else _forecast_draws(fc, var)
        band, med, keep = _forecast_frame(fc, np.asarray(arr), levels, group,
                                          transform=transform)
        obs_df = None
        if obs is True:
            obs_df = _forecast_observed(fc, obs_model, group, n_fitted=n_fitted,
                                        transform=transform)
        band, med, obs_df = _window(band, med, obs_df, dates)
        band, med = _drop_incomplete(band, med)
        p = _ribbon_plot(band, med, palette, levels, ylab, xlab or "", hline,
                         facet=len(keep) if len(keep) > 1 else 0, title=title,
                         obs=obs_df, obs_kind=obs_kind, step=step)
        p = _log_scale(p, log)
        p = _date_axis(p, date_breaks, date_format,
                       _span_days(med, band), len(keep))
        return _maybe_save(p, save, default_name, n_panels=len(keep))

    if _is_multiregion(idata, var):
        if data is None:
            raise ValueError(
                f"{var!r} has a 'region' dimension (a multilevel fit), so the "
                "MultilevelData you fitted is needed to map each region's columns "
                "back to its own dates. Pass data=<your prepare_panel result> "
                "(and optionally group='Italy' for a single region)."
            )
        band, med, keep = _region_frame(idata, var, data, levels, group,
                                        transform=transform, draws=draws)
        obs_df = None
        if obs is True:  # sentinel: take the counts from the panel
            obs_df = _observed_frame(data, group, obs_model=obs_model,
                                     transform=transform)
        elif obs is not None:
            raise ValueError(
                "for a multi-region fit the observed counts come from `data` "
                "(one series per region, each on its own dates), so an "
                "`observed=` array is ambiguous here. Drop it -- plot_obs(idata, "
                "data=fit) already overlays the right counts -- or pass "
                "group='<region>' to plot a single region."
            )
        band, med, obs_df = _window(band, med, obs_df, dates)
        band, med = _drop_incomplete(band, med)
        p = _ribbon_plot(band, med, palette, levels, ylab, xlab or "", hline,
                         facet=len(keep) if len(keep) > 1 else 0, title=title,
                         obs=obs_df, obs_kind=obs_kind, step=step)
        p = _log_scale(p, log)
        p = _date_axis(p, date_breaks, date_format,
                       _span_days(med, band), len(keep))
        return _maybe_save(p, save, default_name, n_panels=len(keep))

    arr, _ = _draws(idata, var)
    if draws is not None:
        arr = np.asarray(draws)
    if transform is not None:
        arr = transform(arr, 0)
    if x is None:
        x = np.arange(arr.shape[-1])
    band, med = _interval_frame(arr, x, levels), _median_frame(arr, x)
    band, med = _drop_incomplete(band, med)
    obs_df = None
    if obs is not None:
        o = np.asarray(obs, dtype=float)
        if transform is not None:
            # the data get the same cumulative/by_100k treatment as the bands
            o = transform(o[None, :], 0)[0]
        obs_df = pd.DataFrame({"x": x[: len(o)], "obs": o})
        obs_df = obs_df[np.isfinite(obs_df["obs"])]
    band, med, obs_df = _window(band, med, obs_df, dates)
    p = _ribbon_plot(band, med, palette, levels, ylab, xlab or "Day", hline,
                     facet=False, title=title, obs=obs_df, obs_kind=obs_kind,
                     step=step)
    p = _log_scale(p, log)
    p = _date_axis(p, date_breaks, date_format, _span_days(med, band))
    return _maybe_save(p, save, default_name)


# --------------------------------------------------------------------------
# Public series plots
# --------------------------------------------------------------------------


def plot_rt(idata, data=None, group=None, groups=None, levels=(30, 60, 90),
            x=None, xlab=None, save=True, title=None, dates=None, log=False,
            smooth=None, step=False, date_breaks=None, date_format=None):
    """Reproduction numbers: posterior median and credible bands.

    For a multilevel fit pass ``data`` (the ``prepare_panel`` result) to get one
    panel per region on real dates; ``group="Italy"`` restricts to one region.
    """
    return _series_plot(idata, "Rt", data, _as_groups(group, groups), levels, x, xlab, "$R_t$",
                        _GREENS, 1.0, save, "rt", title=title, dates=dates,
                        log=log, smooth=smooth, step=step,
                        date_breaks=date_breaks, date_format=date_format)


def plot_infections(idata, data=None, group=None, groups=None,
                    levels=(30, 60, 90), x=None, xlab=None, save=True,
                    title=None, dates=None, log=False, cumulative=False,
                    smooth=None, by_100k=False, date_breaks=None,
                    date_format=None):
    """Latent daily infections: posterior median and credible bands."""
    return _series_plot(idata, "infections", data, _as_groups(group, groups), levels, x, xlab,
                        "Infections", _BLUES, None, save, "infections",
                        title=title, dates=dates, log=log,
                        cumulative=cumulative, smooth=smooth, by_100k=by_100k,
                        date_breaks=date_breaks, date_format=date_format)


def plot_obs(idata, observed=None, data=None, group=None, groups=None, levels=(30, 60, 90), x=None,
             series=None, xlab=None, ylab="Daily deaths", save=True, title=None,
             dates=None, log=False, cumulative=False, smooth=None,
             by_100k=False, bar=False, obs_model=None, predictive=True,
             n_fitted=None, date_breaks=None, date_format=None):
    """Posterior predictive observations with the observed counts overlaid.

    Reads ``E_obs`` (single-population) or ``E_deaths`` (multilevel), whichever
    the fitted model defines. For a multilevel fit pass ``data``; the observed
    counts are then taken from ``data.deaths`` and ``observed`` is not needed.

    Parameters
    ----------
    predictive : bool, default True
        Band the posterior **predictive** -- draws pushed through the
        observation family -- as R's ``plot_obs`` does. The alternative,
        ``False``, bands the posterior of the *expected* count, which excludes
        observation noise and is several times too narrow to compare against the
        data plotted over it. Drawing needs the family, which
        :func:`~epidemia.fit_epidemia` records on the fit; for a fit made before
        that, pass ``obs_model=`` and it is taken from there.
    """
    group = _as_groups(group, groups)
    if _is_forecast(idata):
        # A Forecast already carries predictive draws per series, so nothing
        # needs to be resolved out of a posterior or drawn through a family.
        names = sorted(idata.predicted)
        if series is None:
            if len(names) != 1:
                raise ValueError(
                    f"this forecast has {len(names)} series ({', '.join(names)}); "
                    "pass series= to say which to plot"
                )
            series = names[0]
        elif series not in names:
            raise ValueError(f"no series named {series!r}; have {names}")
        draws = _forecast_draws(idata, series, series, predictive=predictive)
        return _series_plot(idata, series, data, group, levels, x, xlab, ylab,
                            _BLUES, None, save, "obs", title=title,
                            obs=True if obs_model is not None else None,
                            obs_kind="col" if bar else "point",
                            dates=dates, log=log, cumulative=cumulative,
                            smooth=smooth, by_100k=by_100k,
                            obs_model=obs_model, draws=draws,
                            n_fitted=n_fitted, date_breaks=date_breaks,
                            date_format=date_format)

    var = _pick_obs_var(idata, series)
    obs = observed
    if _is_multiregion(idata, var) and obs is None:
        obs = True  # taken from data.deaths inside _series_plot
    # R picks the layer from the flag -- `layer_fun <- if (bar) geom_bar else
    # geom_point` (R/plots_epi.R:244) -- rather than from the model shape. The
    # old test here compared against "bar", a value `kind` never takes, so the
    # branch was dead: bar=False still drew columns and a single-population fit
    # could never draw them.
    kind = "col" if bar else "point"
    draws = None
    if predictive:
        name = series if series is not None else var[2:]
        draws = _predictive_draws(idata, var, name, obs_model)
        if draws is None:
            import warnings
            warnings.warn(
                f"cannot draw the posterior predictive for {name!r}: the "
                "observation family is not recorded on this fit and no "
                "obs_model= was given. Falling back to banding the expected "
                "count, whose interval excludes observation noise and is "
                "therefore too narrow to compare against the plotted data. "
                "Pass obs_model=<the ObsModel> or predictive=False to silence.",
                stacklevel=2,
            )
    return _series_plot(idata, var, data, group, levels, x, xlab, ylab,
                        _BLUES, None, save, "obs", title=title, obs=obs,
                        obs_kind=kind, dates=dates, log=log,
                        cumulative=cumulative, smooth=smooth, by_100k=by_100k,
                        obs_model=obs_model, draws=draws, n_fitted=n_fitted,
                        date_breaks=date_breaks, date_format=date_format)


# --------------------------------------------------------------------------
# Effect-size plots (multilevel)
# --------------------------------------------------------------------------


def _forest(df, xlab, title, hline=0.0):
    """A bayesplot-style interval plot: thick inner band, thin outer, point median.

    R's ``bayesplot::mcmc_intervals`` draws two nested credible intervals per
    parameter. A single line with a dot -- what this used to draw -- throws away
    the shape of the posterior and reads as a frequentist error bar.
    """
    has_inner = {"lo_in", "hi_in"} <= set(df.columns)
    p = ggplot(df, aes("term", "median"))
    p = p + geom_hline(yintercept=hline, linetype="dotted", color="#999999",
                       size=0.5)
    if has_inner:
        p = p + geom_pointrange(aes(ymin="lo", ymax="hi"), size=0.22,
                                color=_BLUES[60], fatten=0)
        p = p + geom_pointrange(aes(ymin="lo_in", ymax="hi_in"), size=0.75,
                                color=_BLUES[30], fatten=0)
    else:
        p = p + geom_pointrange(aes(ymin="lo", ymax="hi"), size=0.45,
                                color=_BLUES[30], fatten=0)
    p = (
        p
        + geom_point(size=2.4, color="white")
        + geom_point(size=1.5, color=COLORS["median"])
        + coord_flip()
        + labs(x="", y=xlab, title=title)
        # No tilt: after coord_flip the x axis carries the VALUE, and tilting a
        # numeric axis is noise. The 45-degree default exists for dates.
        + theme_epidemia(rotate_x=0)
    )
    return p

def plot_effects(idata, group=None, labels=None, levels=(50, 90), save=True,
                 title=None):
    """Forest plot of the NPI effects on the logit-``R_t`` scale.

    ``group=None`` shows the **global** (fixed) effects :math:`\\beta_k` — the
    average effect across regions. ``group="Italy"`` shows that region's total
    effect :math:`\\beta_k + b^{(m)}_k`, which is what actually drives its
    :math:`R_t` and can differ from the global value under partial pooling.
    """
    inner, outer = (sorted(levels)[:2] if isinstance(levels, (list, tuple))
                    else (levels, levels))
    lo_q, hi_q = (100 - outer) / 2, 100 - (100 - outer) / 2
    lo_i, hi_i = (100 - inner) / 2, 100 - (100 - inner) / 2
    beta = np.asarray(idata.posterior["beta"].stack(s=("chain", "draw")))  # (K, S)
    npis = [str(v) for v in idata.posterior.coords["npi"].values]
    mat = beta
    if group is not None:
        b = np.asarray(
            idata.posterior["b"].sel(region=group).stack(s=("chain", "draw"))
        )  # (K, S)
        mat = beta + b
    terms = list(labels) if labels is not None else npis
    df = pd.DataFrame({
        "term": pd.Categorical(terms, categories=list(reversed(terms))),
        "median": np.median(mat, axis=1),
        "lo": np.percentile(mat, lo_q, axis=1),
        "hi": np.percentile(mat, hi_q, axis=1),
        "lo_in": np.percentile(mat, lo_i, axis=1),
        "hi_in": np.percentile(mat, hi_i, axis=1),
    })
    if title is None:
        title = ("Global NPI effects $\\beta_k$" if group is None
                 else f"{group}-specific effects $\\beta_k + b^{{({group})}}_k$")
    p = _forest(df, "Effect on logit $R_t$", title)
    return _maybe_save(p, save, "effects" if group is None else f"effects_{group}")


def plot_percent_effects(idata, config, data=None, group=None, labels=None,
                         levels=90, save=True, title=None):
    """Effect of each measure as a **percent reduction in transmission**.

    The readable counterpart of :func:`plot_effects`: instead of a coefficient on
    the logit scale, each measure is shown as the percent by which it cuts
    ``R_t`` relative to that region's no-measures baseline (so 90 means "``R_t``
    is a tenth of what it would otherwise be"). See
    :func:`epidemia.effect_table` for how it is computed and why it is per-region
    rather than one global number.

    Pass ``data`` to grey out measures a region never actually enacted: those
    percentages are counterfactual extrapolations from the pooled prior, not
    measured effects, and they should not be read the same way.
    """
    from .multilevel import effect_table

    tab = effect_table(idata, config, data=data, group=group, levels=levels)
    tab = tab[tab["kind"] == "pct"].copy()
    if labels is not None:
        npis = [str(n) for n in idata.posterior.coords["npi"].values]
        rename = dict(zip(npis, labels))
        tab["term"] = tab["term"].map(lambda t: rename.get(t, t))
    order = list(dict.fromkeys(tab["term"]))
    tab["term"] = pd.Categorical(tab["term"], categories=list(reversed(order)))
    n_panels = tab["region"].nunique()

    show_enacted = data is not None and tab["enacted"].notna().any()
    if show_enacted:
        tab["evidence"] = np.where(
            tab["enacted"].fillna(True).astype(bool), "enacted here",
            "never enacted here (counterfactual)")
        p = (
            ggplot(tab, aes("term", "median", color="evidence"))
            + scale_color_manual(
                values={"enacted here": "#238b45",
                        "never enacted here (counterfactual)": "#bbbbbb"},
                name="")
        )
    else:
        p = ggplot(tab, aes("term", "median"))
    p = (
        p
        + geom_hline(yintercept=0.0, linetype="dotted", color="#555555")
        + geom_pointrange(aes(ymin="lo", ymax="hi"))
        + coord_flip()
        + labs(x="", y="Reduction in $R_t$ (%)",
               title=title or "Effect of each measure on transmission")
        + theme_epidemia()
    )
    if n_panels > 1:
        p = p + facet_wrap("region", ncol=_FACET_NCOL)
        p = _size_for_panels(p, n_panels)
    return _maybe_save(p, save, "percent-effects" if group is None
                       else f"percent-effects_{group}", n_panels=n_panels)


def plot_region_effects(idata, npi, levels=90, save=True, title=None):
    """Forest plot of one NPI's **per-region** total effect ``beta_k + b[m, k]``.

    This is the plot that answers "is the effect zero *here*?" — the global
    ``beta_k`` alone does not, because partial pooling splits each region's effect
    between the shared ``beta_k`` and its own deviation ``b[m, k]``.
    """
    inner, outer = (sorted(levels)[:2] if isinstance(levels, (list, tuple))
                    else (levels, levels))
    lo_q, hi_q = (100 - outer) / 2, 100 - (100 - outer) / 2
    lo_i, hi_i = (100 - inner) / 2, 100 - (100 - inner) / 2
    post = idata.posterior
    beta = np.asarray(post["beta"].sel(npi=npi).stack(s=("chain", "draw")))
    regions = [str(r) for r in post.coords["region"].values]
    rows = []
    for r in regions:
        b = np.asarray(post["b"].sel(region=r, npi=npi).stack(s=("chain", "draw")))
        e = beta + b
        rows.append({"term": r, "median": np.median(e),
                     "lo": np.percentile(e, lo_q), "hi": np.percentile(e, hi_q)})
    df = pd.DataFrame(rows)
    df["term"] = pd.Categorical(df["term"], categories=list(reversed(regions)))
    p = _forest(df, f"{npi} effect on logit $R_t$  ($\\beta + b^{{(m)}}$)",
                title or f"Per-region effect of {npi}")
    return _maybe_save(p, save, f"region_effects_{npi}")


# --------------------------------------------------------------------------
# Spaghetti (per-draw trajectory) plots
# --------------------------------------------------------------------------
#
# The ribbon plots above collapse the posterior to pointwise quantiles, which
# throws away the *shape* of an individual path: a 95% band is the envelope of
# many trajectories, and no single draw need look like its median. R's
# spaghetti_rt() / spaghetti_infections() / spaghetti_obs() overlay the paths
# themselves, so autocorrelation and the plausibility of individual epidemic
# curves stay visible. These mirror them.

# darkest shade of each ribbon palette, so a spaghetti plot sits next to the
# corresponding plot_*() without a colour clash
_RT_PATH_COLOR = _GREENS[30]
_OBS_PATH_COLOR = _BLUES[30]


def _sample_draw_index(n_draws, draws, seed):
    """Indices of the draws to overlay -- reproducible, sorted, without replacement.

    Asking for more paths than the posterior holds is *not* an error: R's default
    is ``min(500, posterior_sample_size(object))``, so the request is a cap, and a
    short (or thinned) chain should still plot rather than blow up.
    """
    if draws is None:
        k = n_draws
    else:
        k = int(draws)
        if k < 1:
            raise ValueError(f"draws must be >= 1, got {draws}")
        k = min(k, n_draws)
    # A seeded Generator (not the global RNG) so re-running the script, or
    # regenerating a figure for a paper, redraws the same set of paths.
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_draws, size=k, replace=False))


def _check_alpha(alpha):
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha


def _path_frame(draws, x, idx):
    """Long frame of the individual trajectories ``draws[idx]``: one row per point.

    ``draw`` is stringified because it is a *grouping* aesthetic, not a
    continuous one -- a numeric column would tempt plotnine into treating the
    draw index as a scale.
    """
    sub = np.asarray(draws)[idx]  # (k, T)
    k, t = sub.shape
    return pd.DataFrame({
        "x": np.tile(np.asarray(x), k),
        "value": sub.reshape(-1),
        "draw": np.repeat([str(i) for i in idx], t),
    })


def _region_path_frame(idata, var, data, idx, group=None, draws=None,
                       transform=None):
    """Per-region path + median frames on real dates, padded days dropped.

    The same draw indices are used in every region: a draw is a joint sample over
    all regions, so pairing region A's draw 3 with region B's draw 7 would show a
    combination the posterior never produced.
    """
    arr, dims = _draws(idata, var)  # (draws, region, time)
    if draws is not None:
        arr = np.asarray(draws)     # predictive draws, same shape
    if dims[:1] != ("region",):
        raise ValueError(f"expected a region dim on {var!r}, got dims {dims}")
    regions = [str(r) for r in idata.posterior.coords["region"].values]
    want = _as_groups(group, None)
    keep = regions if want is None else want
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    paths, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(data.lengths[data.regions.index(r)])
        x = pd.to_datetime(data.dates[data.regions.index(r)])[:n]
        d = arr[:, m, :n]
        if transform is not None:
            d = transform(d, data.regions.index(r))
        pf = _path_frame(d, x, idx)
        pf["region"] = r
        paths.append(pf)
        md = _median_frame(d, x)  # median over ALL draws, as in R
        md["region"] = r
        meds.append(md)
    return pd.concat(paths, ignore_index=True), pd.concat(meds, ignore_index=True), keep



def _forecast_path_frame(fc, arr, idx, group=None, transform=None):
    """Per-region path + median frames from a Forecast, padding dropped."""
    regions = [str(r) for r in fc.regions]
    want = _as_groups(group, None)
    keep = regions if want is None else want
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")
    paths, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(fc.lengths[m])
        x = pd.to_datetime(fc.dates[m])[:n]
        d = arr[:, m, :n]
        if transform is not None:
            d = transform(d, m)
        pf = _path_frame(d, x, idx)
        pf["region"] = r
        paths.append(pf)
        mf = _median_frame(d, x)
        mf["region"] = r
        meds.append(mf)
    return (pd.concat(paths, ignore_index=True),
            pd.concat(meds, ignore_index=True), keep)


def _spaghetti_plot(paths, med, color, alpha, ylab, xlab, hline=None, facet=0,
                    title=None, obs=None, obs_kind="point"):
    """``facet`` is the number of region panels (0/1 => no faceting)."""
    p = (
        ggplot()
        + geom_line(paths, aes("x", "value", group="draw"), color=color,
                    alpha=alpha, size=0.35)
    )
    if obs is not None:
        has_period = "period" in getattr(obs, "columns", [])
        pal = {"In-sample": COLORS["in_sample"],
               "Out-of-sample": COLORS["out_of_sample"]}
        if has_period:
            if obs_kind == "col":
                p = (p + geom_col(obs, aes("x", "obs", fill="period"),
                                  alpha=0.55, width=0.75)
                       + scale_fill_manual(values=pal, name=""))
            else:
                p = (p + geom_point(obs, aes("x", "obs", color="period"),
                                    size=0.9, alpha=0.75, stroke=0)
                       + scale_color_manual(values=pal, name=""))
        elif obs_kind == "col":
            p = p + geom_col(obs, aes("x", "obs"), fill=COLORS["observed"],
                             alpha=0.45, width=0.75)
        else:
            p = p + geom_point(obs, aes("x", "obs"), color=COLORS["observed"],
                               size=0.9, alpha=0.75, stroke=0)
    return p


def _spaghetti_series(idata, var, data, group, draws, alpha, seed, x, xlab, ylab,
                      color, hline, save, default_name, title=None, obs=None,
                      obs_kind="point", obs_model=None, draws_override=None,
                      dates=None, log=False, smooth=None, cumulative=False,
                      by_100k=False, step=False):
    """Dispatch: multi-region (facet on real dates) vs single-population."""
    _check_alpha(alpha)
    transform = _draw_transform(cumulative, smooth, by_100k,
                                getattr(data, "pops", None))
    if _is_forecast(idata):
        # A Forecast carries its own draws, dates and lengths -- the ribbon
        # plots have handled one since they gained _is_forecast; the spaghetti
        # path never consulted it and went straight to idata.posterior.
        fc = idata
        arr = (np.asarray(draws_override) if draws_override is not None
               else _forecast_draws(fc, var))
        idx = _sample_draw_index(arr.shape[0], draws, seed)
        paths, med, keep = _forecast_path_frame(fc, arr, idx,
                                                _as_groups(group, None),
                                                transform=transform)
        obs_df = (_forecast_observed(fc, obs_model, group)
                  if obs is True and obs_model is not None else None)
        paths, med, obs_df = _window(paths, med, obs_df, dates)
        p = _spaghetti_plot(paths, med, color, alpha, ylab, xlab or "", hline,
                            facet=len(keep) if len(keep) > 1 else 0, title=title,
                            obs=obs_df, obs_kind=obs_kind)
        p = _log_scale(p, log)
        return _maybe_save(p, save, default_name, n_panels=len(keep))

    arr, _ = _draws(idata, var)
    if draws_override is not None:
        arr = np.asarray(draws_override)
    idx = _sample_draw_index(arr.shape[0], draws, seed)

    if _is_multiregion(idata, var):
        if data is None:
            raise ValueError(
                f"{var!r} has a 'region' dimension (a multilevel fit), so the "
                "MultilevelData you fitted is needed to map each region's columns "
                "back to its own dates. Pass data=<your prepare_panel result> "
                "(and optionally group='Italy' for a single region)."
            )
        paths, med, keep = _region_path_frame(idata, var, data, idx,
                                              _as_groups(group, None),
                                              draws=draws_override,
                                              transform=transform)
        obs_df = None
        if obs is True:  # sentinel: take the counts from the panel
            obs_df = _observed_frame(data, group, obs_model=obs_model,
                                     transform=transform)
        elif obs is not None:
            raise ValueError(
                "for a multi-region fit the observed counts come from `data` "
                "(one series per region, each on its own dates), so an "
                "`observed=` array is ambiguous here. Drop it -- "
                "spaghetti_obs(idata, data=fit) already overlays the right "
                "counts -- or pass group='<region>' to plot a single region."
            )
        p = _spaghetti_plot(paths, med, color, alpha, ylab, xlab or "", hline,
                            facet=len(keep) if len(keep) > 1 else 0, title=title,
                            obs=obs_df, obs_kind=obs_kind)
        return _maybe_save(p, save, default_name, n_panels=len(keep))

    if x is None:
        x = np.arange(arr.shape[-1])
    paths, med = _path_frame(arr, x, idx), _median_frame(arr, x)
    obs_df = None
    if obs is not None:
        o = np.asarray(obs, dtype=float)
        obs_df = pd.DataFrame({"x": x, "obs": o})
        obs_df = obs_df[np.isfinite(obs_df["obs"])]
    p = _spaghetti_plot(paths, med, color, alpha, ylab, xlab or "Day", hline,
                        facet=0, title=title, obs=obs_df, obs_kind=obs_kind)
    return _maybe_save(p, save, default_name)


def spaghetti_rt(idata, data=None, group=None, groups=None, draws=50, alpha=0.3,
                 seed=0, x=None, xlab=None, save=True, title=None, region=None,
                 dates=None, log=False, smooth=None, step=False,
                date_breaks=None, date_format=None):
    """Reproduction numbers: ``draws`` individual posterior paths, median on top.

    The counterpart of :func:`plot_rt` for looking at trajectories rather than
    pointwise quantiles. ``draws`` caps the number of paths (fewer are drawn if
    the posterior is smaller) and ``seed`` fixes which ones, so the figure is
    reproducible. ``alpha`` is the per-path transparency: lower it when the
    bundle is dense.

    ``region`` is an alias for ``group``, kept for symmetry with R's ``groups``.
    """
    return _spaghetti_series(idata, "Rt", data,
                             _as_groups(group if group is not None else region,
                                        groups),
                             draws, alpha, seed, x, xlab, "$R_t$", _RT_PATH_COLOR,
                             1.0, save, "spaghetti-rt", title=title,
                             dates=dates, log=log, smooth=smooth, step=step)


def spaghetti_infections(idata, data=None, group=None, groups=None, draws=50,
                         alpha=0.3, seed=0, x=None, xlab=None, save=True,
                         title=None, region=None, dates=None, log=False,
                         smooth=None, cumulative=False, by_100k=False):
    """Latent daily infections: ``draws`` individual posterior paths, median on top."""
    return _spaghetti_series(idata, "infections", data,
                             _as_groups(group if group is not None else region,
                                        groups),
                             draws, alpha, seed, x, xlab, "Infections",
                             _OBS_PATH_COLOR, None, save, "spaghetti-infections",
                             title=title, dates=dates, log=log, smooth=smooth,
                             cumulative=cumulative, by_100k=by_100k)


def spaghetti_obs(idata, observed=None, data=None, group=None, draws=50, alpha=0.3,
                  series=None,
                  seed=0, x=None, xlab=None, ylab="Daily deaths", save=True,
                  title=None, region=None, obs_model=None, predictive=True,
                  groups=None, dates=None, log=False, smooth=None,
                  cumulative=False, by_100k=False, bar=False):
    """Expected observations as individual paths, with the observed counts overlaid.

    Reads ``E_obs`` (single-population) or ``E_deaths`` (multilevel), whichever
    the fitted model defines; raises :class:`KeyError` if neither is present. For
    a multilevel fit pass ``data`` and the counts are taken from ``data.deaths``;
    a :class:`~epidemia.core.PanelData` carries no counts, so pass
    ``obs_model=`` (the :class:`~epidemia.core.ObsModel` for this series) there.
    """
    if _is_forecast(idata):
        # Resolve the series against the Forecast's own dict; _pick_obs_var
        # reads idata.posterior, which a Forecast does not have.
        names = sorted(idata.predicted)
        if series is None:
            if len(names) != 1:
                raise ValueError(
                    f"this forecast has {len(names)} series ({', '.join(names)}); "
                    "pass series= to say which to plot")
            series = names[0]
        elif series not in names:
            raise ValueError(f"no series named {series!r}; have {names}")
        var = series
    else:
        var = _pick_obs_var(idata, series)
    grp = _as_groups(group if group is not None else region, groups)
    obs = observed
    if _is_multiregion(idata, var) and obs is None:
        obs = True  # taken from data.deaths inside _spaghetti_series
    kind = "col" if bar else "point"
    return _spaghetti_series(idata, var, data, grp, draws, alpha, seed, x, xlab, ylab,
                             _OBS_PATH_COLOR, None, save, "spaghetti-obs", title=title,
                             obs=obs, obs_kind=kind, obs_model=obs_model,
                             draws_override=(
                                 _predictive_draws(
                                     idata, var,
                                     series if series is not None else var[2:],
                                     obs_model)
                                 if predictive and not _is_forecast(idata)
                                 else None),
                             dates=dates, log=log, smooth=smooth,
                             cumulative=cumulative, by_100k=by_100k)


# --------------------------------------------------------------------------
# Latent series that are computed rather than stored
# --------------------------------------------------------------------------


def plot_infectious(idata, config, data=None, group=None, groups=None,
                    levels=(30, 60, 90), x=None, xlab=None,
                    ylab="Infectiousness", save=True,
                    title=None, dates=None, log=False, smooth=None):
    """Total infectiousness over time -- R's ``plot_infectious``.

    Bands :func:`~epidemia.postprocess.posterior_infectious`, which is the
    generation-weighted sum of past infections divided by ``max(gen)``, exactly
    as R's ``epidemia_pp_base.stan`` computes it. Unlike ``infections`` this is
    not stored in the trace, so it is recomputed from the posterior here.
    """
    from .postprocess import posterior_infectious

    draws = posterior_infectious(idata, config)
    return _series_plot(idata, "infections", data, _as_groups(group, groups), levels, x, xlab, ylab,
                        _BLUES, None, save, "infectious", title=title,
                        dates=dates, log=log, smooth=smooth, draws=draws)


def plot_linpred(idata, panel, config, series=None, obs_models=None, data=None,
                 group=None, groups=None, levels=(30, 60, 90), x=None, xlab=None,
                 ylab=None, save=True, title=None, dates=None, transform=False,
                 smooth=None):
    """The linear predictor over time -- R's ``plot_linpred``.

    Bands :func:`~epidemia.postprocess.posterior_linpred`. ``series=None`` gives
    the ``R_t`` predictor; naming a series gives that observation model's
    ascertainment predictor. ``transform=True`` applies the inverse link, as in
    R.
    """
    from .postprocess import posterior_linpred

    draws = posterior_linpred(idata, panel, config, series=series,
                              obs_models=obs_models, transform=transform)
    if ylab is None:
        what = "R_t" if series is None else series
        ylab = f"{what} predictor" + (" (transformed)" if transform else "")
    template = "Rt_unadj" if "Rt_unadj" in idata.posterior else "Rt"
    return _series_plot(idata, template, data if data is not None else panel,
                        _as_groups(group, groups), levels, x, xlab, ylab,
                        _GREENS, None, save,
                        "linpred", title=title, dates=dates, smooth=smooth,
                        draws=draws)


# --------------------------------------------------------------------------
# Forecast scoring plots
# --------------------------------------------------------------------------


def plot_coverage(y, draws, group=None, date=None, levels=(50, 95),
                  period=None, by_group=False, by_unseen=None, save=True,
                  title=None):
    """Empirical coverage of the credible intervals -- R's ``plot_coverage``.

    A bar chart of the share of observations that fell inside each interval. A
    well-calibrated 95% interval covers ~95%; bars far below that mean the
    forecast is overconfident.

    Parameters
    ----------
    y, draws, group, date, levels
        As :func:`epidemia.scoring.posterior_coverage`: ``draws`` has one row
        per observation.
    period : str, optional
        Bucket the dates before averaging, e.g. ``"W"`` or ``"M"`` (any pandas
        offset alias). R takes a ``cut()`` spec; the intent is the same.
    by_group : bool
        One panel per group.
    by_unseen : bool array, optional
        Mark which observations were NOT used in fitting, and facet on it. R
        infers this by joining against the fit window; here it is supplied
        directly, because the scoring functions take arrays rather than a model.
    """
    from .scoring import posterior_coverage

    cov = posterior_coverage(y, draws, group=group, date=date, levels=levels)
    cols = ["tag"]
    if period is not None:
        cov = cov.copy()
        cov["period"] = pd.to_datetime(cov["date"]).dt.to_period(period).astype(str)
        cols.append("period")
    if by_group:
        cols.append("group")
    if by_unseen is not None:
        cov = cov.copy()
        flag = np.asarray(by_unseen, dtype=bool)
        n_obs = len(np.asarray(y).reshape(-1))
        if flag.size != n_obs:
            raise ValueError(
                f"by_unseen must have one entry per observation ({n_obs}), "
                f"got {flag.size}"
            )
        # posterior_coverage stacks one row per (observation, level)
        cov["unseen"] = np.where(np.tile(flag, len(levels)), "Unseen", "Seen")
        cols.append("unseen")

    df = cov.groupby(cols, as_index=False)["in_ci"].mean().rename(
        columns={"in_ci": "value"})

    xvar = "period" if period is not None else "tag"
    p = (
        ggplot(df, aes(xvar, "value", fill="tag"))
        + geom_col(position="dodge")
        + geom_hline(yintercept=[lv / 100 for lv in levels],
                     linetype="dotted", color="#555555")
        + labs(x="Credible interval" if period is None else "Period",
               y="Mean coverage", fill="", title=title)
        + theme_epidemia(rotate_x=0 if period is None else 45)
    )
    facets = [c for c in ("group", "unseen") if c in cols]
    n_panels = 1
    if facets:
        p = p + facet_wrap("~" + " + ".join(facets))
        n_panels = int(df.groupby(facets).ngroups)
    p = _size_for_panels(p, n_panels)
    return _maybe_save(p, save, "coverage", n_panels=n_panels)


def plot_metrics(y, draws, group=None, date=None, metrics=None, by_unseen=None,
                 save=True, title=None):
    """CRPS and absolute error over time -- R's ``plot_metrics``.

    One line per metric, faceted by group. ``by_unseen`` colours the in-sample
    and out-of-sample parts differently, as R does.
    """
    from .scoring import posterior_metrics

    df = posterior_metrics(y, draws, group=group, date=date, metrics=metrics)
    value_cols = [c for c in ("crps", "mean_abs_error", "median_abs_error")
                  if c in df.columns]
    long = df.melt(id_vars=[c for c in ("group", "date") if c in df.columns],
                   value_vars=value_cols, var_name="metric", value_name="value")

    color = None
    if by_unseen is not None:
        flag = np.asarray(by_unseen, dtype=bool)
        if flag.size != len(df):
            raise ValueError(
                f"by_unseen must have one entry per observation ({len(df)}), "
                f"got {flag.size}"
            )
        seen = pd.Series(np.where(flag, "Unseen", "Seen"), index=df.index)
        long = long.merge(
            pd.DataFrame({"group": df.get("group"), "date": df.get("date"),
                          "unseen": seen}),
            on=[c for c in ("group", "date") if c in df.columns], how="left")
        color = "unseen"

    mapping = aes("date", "value", linetype="metric",
                  **({"color": color} if color else {}))
    p = ggplot(long, mapping) + geom_line(alpha=0.8, size=0.8)
    if color:
        p = p + scale_color_manual(values=["#2171b5", "#b5451f"])
    p = p + labs(x="Date", y="Value", linetype="Metric",
                 color="" if color else None, title=title) + theme_epidemia()
    n_panels = 1
    if "group" in long.columns and long["group"].nunique() > 1:
        p = p + facet_wrap("~group", scales="free_y")
        n_panels = int(long["group"].nunique())
        p = _size_for_panels(p, n_panels)
    return _maybe_save(p, save, "metrics", n_panels=n_panels)


# --------------------------------------------------------------------------
# Parameter plots (R's plot.epimodel / pairs.epimodel)
# --------------------------------------------------------------------------


def plot_intervals(idata, pars=None, regex=None, series=None, par_types=None,
                   levels=(50, 90), save=True, title=None):
    """Credible intervals for individual parameters -- R's ``plot.epimodel``.

    A forest plot with one row per parameter. Selection mirrors
    :func:`epidemia.postprocess.extract_samples`, so ``par_types="fixed"`` gives
    the fixed effects and ``par_types="seeds"`` the seeds, as R's
    ``par_types`` does.
    """
    from .postprocess import extract_samples

    draws = extract_samples(idata, pars=pars, regex=regex, series=series,
                            par_types=par_types)
    if not len(draws.columns):
        raise ValueError("no parameters matched the selection")

    inner, outer = sorted(levels)[:2] if len(levels) >= 2 else (levels[0], levels[0])
    rows = []
    for col in draws.columns:
        v = draws[col].to_numpy(dtype=float)
        rows.append({
            "term": col,
            "median": np.median(v),
            "lo": np.percentile(v, (100 - outer) / 2),
            "hi": np.percentile(v, 100 - (100 - outer) / 2),
            "lo_in": np.percentile(v, (100 - inner) / 2),
            "hi_in": np.percentile(v, 100 - (100 - inner) / 2),
        })
    df = pd.DataFrame(rows)
    order = list(reversed(df["term"].tolist()))
    df["term"] = pd.Categorical(df["term"], categories=order)

    p = (
        ggplot(df, aes("term", "median"))
        + geom_hline(yintercept=0.0, linetype="dotted", color="#555555")
        + geom_pointrange(aes(ymin="lo", ymax="hi"), size=0.25, color="#6baed6")
        + geom_pointrange(aes(ymin="lo_in", ymax="hi_in"), size=0.45,
                          color="#2171b5")
        + coord_flip()
        + labs(x="", y="Value", title=title)
        + theme_epidemia(rotate_x=0)
    )
    n = max(1, int(np.ceil(len(df) / 6)))
    p = p + theme(figure_size=(8.0, 1.2 + 0.32 * len(df)))
    return _maybe_save(p, save, "intervals", n_panels=1)


def pairs_plot(idata, pars=None, regex=None, series=None, par_types=None,
               draws=500, seed=0, save=True, title=None):
    """Pairwise scatter of a few parameters -- R's ``pairs.epimodel``.

    Useful for spotting the ridges and funnels that
    :func:`epidemia.sampler_diagnostics` reports as divergences: a banana
    between a scale and its increments is exactly what a divergent transition
    is complaining about.

    Keep the selection small -- the panel count grows as the square.
    """
    from .postprocess import extract_samples

    df = extract_samples(idata, pars=pars, regex=regex, series=series,
                         par_types=par_types)
    cols = list(df.columns)
    if not 2 <= len(cols) <= 8:
        raise ValueError(
            f"pairs_plot wants between 2 and 8 parameters, got {len(cols)}; "
            "narrow the selection with pars=, regex= or par_types="
        )
    rng = np.random.default_rng(seed)
    if len(df) > draws:
        df = df.iloc[np.sort(rng.choice(len(df), size=int(draws), replace=False))]

    long = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            long.append(pd.DataFrame({
                "x": df[b].to_numpy(dtype=float),
                "y": df[a].to_numpy(dtype=float),
                "panel": f"{a}  vs  {b}",
            }))
    frame = pd.concat(long, ignore_index=True)

    p = (
        ggplot(frame, aes("x", "y"))
        + geom_point(size=0.5, alpha=0.35, color="#2171b5")
        + facet_wrap("panel", scales="free")
        + labs(x="", y="", title=title)
        + theme_epidemia()
    )
    n_panels = frame["panel"].nunique()
    p = _size_for_panels(p, n_panels)
    return _maybe_save(p, save, "pairs", n_panels=n_panels)
