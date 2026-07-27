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

# single-hue sequential ramps (light -> dark), colour-blind friendly
_GREENS = {30: "#238b45", 50: "#238b45", 60: "#74c476", 90: "#c7e9c0", 95: "#c7e9c0"}
_BLUES = {30: "#2171b5", 50: "#2171b5", 60: "#6baed6", 90: "#c6dbef", 95: "#c6dbef"}


def theme_epidemia(base_size: float = 11):
    """A clean, publication-oriented plotnine theme."""
    return theme_minimal(base_size=base_size) + theme(
        figure_size=(8, 4.5),
        dpi=120,
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(color="#e9e9e9", size=0.4),
        axis_title=element_text(size=base_size),
        axis_text=element_text(size=base_size - 1, color="#333333"),
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


def _is_multiregion(idata, var):
    return "region" in idata.posterior[var].dims


def available_series(idata):
    """Names of the observation series present in ``idata``.

    ``build_epidemia_model`` records one ``E_<name>`` per series and the names
    are whatever the user called them, so they cannot be guessed.
    """
    return sorted(
        v[2:] for v in idata.posterior.data_vars if str(v).startswith("E_")
    )


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


def _region_frame(idata, var, data, levels, group=None):
    """Per-region interval + median frames on real dates, padded days dropped.

    ``data`` is the :class:`~epidemia.multilevel.MultilevelData` that was fitted;
    it supplies each region's genuine dates and length, without which column ``t``
    would be a *different date* in every region.
    """
    arr, dims = _draws(idata, var)  # (draws, region, time)
    if dims[:1] != ("region",):
        raise ValueError(f"expected a region dim on {var!r}, got dims {dims}")
    regions = [str(r) for r in idata.posterior.coords["region"].values]
    keep = regions if group is None else [str(group)]
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    bands, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(data.lengths[data.regions.index(r)])
        x = pd.to_datetime(data.dates[data.regions.index(r)])[:n]
        d = arr[:, m, :n]
        b = _interval_frame(d, x, levels)
        b["region"] = r
        bands.append(b)
        md = _median_frame(d, x)
        md["region"] = r
        meds.append(md)
    return pd.concat(bands, ignore_index=True), pd.concat(meds, ignore_index=True), keep


def _observed_frame(data, group=None):
    """Long frame of the genuinely **observed** counts per region on real dates.

    Days ``prepare_panel`` masked out (a missing count) are dropped rather than
    drawn. ``data.deaths`` is zero-filled on those days -- only ``mask`` records
    that the zero is a placeholder -- so reading ``deaths`` alone would plot a
    fabricated zero-death observation the model was never fit to.
    """
    keep = data.regions if group is None else [str(group)]
    rows = []
    for r in keep:
        m = data.regions.index(r)
        n = int(data.lengths[m])
        obs = np.asarray(data.deaths[m, :n], dtype=float)
        obs[~np.asarray(data.mask)[m, :n]] = np.nan  # missing is not a zero
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
                 title=None, obs=None, obs_kind="point"):
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
        if obs_kind == "col":
            p = p + geom_col(obs, aes("x", "obs"), fill="#b2182b", alpha=0.45, width=1.0)
        else:
            p = p + geom_point(obs, aes("x", "obs"), color="#b2182b", size=1.1, alpha=0.8)
    p = p + geom_line(med, aes("x", "median"), color="black", size=0.6) + labs(x=xlab, y=ylab)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dotted", color="#555555")
    if title:
        p = p + labs(title=title)
    p = p + theme_epidemia()
    if facet:
        p = p + facet_wrap("region", scales="free_y", ncol=_FACET_NCOL)
        p = _size_for_panels(p, facet)  # `facet` carries the panel count
    return p


def _series_plot(idata, var, data, group, levels, x, xlab, ylab, palette, hline,
                 save, default_name, title=None, obs=None, obs_kind="point"):
    """Dispatch: multi-region (facet on real dates) vs single-population."""
    if _is_multiregion(idata, var):
        if data is None:
            raise ValueError(
                f"{var!r} has a 'region' dimension (a multilevel fit), so the "
                "MultilevelData you fitted is needed to map each region's columns "
                "back to its own dates. Pass data=<your prepare_panel result> "
                "(and optionally group='Italy' for a single region)."
            )
        band, med, keep = _region_frame(idata, var, data, levels, group)
        obs_df = None
        if obs is True:  # sentinel: take the counts from the panel
            obs_df = _observed_frame(data, group)
        elif obs is not None:
            raise ValueError(
                "for a multi-region fit the observed counts come from `data` "
                "(one series per region, each on its own dates), so an "
                "`observed=` array is ambiguous here. Drop it -- plot_obs(idata, "
                "data=fit) already overlays the right counts -- or pass "
                "group='<region>' to plot a single region."
            )
        p = _ribbon_plot(band, med, palette, levels, ylab, xlab or "", hline,
                         facet=len(keep) if len(keep) > 1 else 0, title=title,
                         obs=obs_df, obs_kind=obs_kind)
        return _maybe_save(p, save, default_name, n_panels=len(keep))

    arr, _ = _draws(idata, var)
    if x is None:
        x = np.arange(arr.shape[-1])
    band, med = _interval_frame(arr, x, levels), _median_frame(arr, x)
    obs_df = None
    if obs is not None:
        o = np.asarray(obs, dtype=float)
        obs_df = pd.DataFrame({"x": x, "obs": o})
        obs_df = obs_df[np.isfinite(obs_df["obs"])]
    p = _ribbon_plot(band, med, palette, levels, ylab, xlab or "Day", hline,
                     facet=False, title=title, obs=obs_df, obs_kind=obs_kind)
    return _maybe_save(p, save, default_name)


# --------------------------------------------------------------------------
# Public series plots
# --------------------------------------------------------------------------


def plot_rt(idata, data=None, group=None, levels=(50, 95), x=None, xlab=None,
            save=True, title=None):
    """Reproduction numbers: posterior median and credible bands.

    For a multilevel fit pass ``data`` (the ``prepare_panel`` result) to get one
    panel per region on real dates; ``group="Italy"`` restricts to one region.
    """
    return _series_plot(idata, "Rt", data, group, levels, x, xlab, "$R_t$",
                        _GREENS, 1.0, save, "rt", title=title)


def plot_infections(idata, data=None, group=None, levels=(50, 95), x=None, xlab=None,
                    save=True, title=None):
    """Latent daily infections: posterior median and credible bands."""
    return _series_plot(idata, "infections", data, group, levels, x, xlab, "Infections",
                        _BLUES, None, save, "infections", title=title)


def plot_obs(idata, observed=None, data=None, group=None, levels=(50, 95), x=None,
             series=None,
             xlab=None, ylab="Daily deaths", save=True, title=None):
    """Expected observations with the observed counts overlaid (a posterior check).

    Reads ``E_obs`` (single-population) or ``E_deaths`` (multilevel), whichever
    the fitted model defines. For a multilevel fit pass ``data``; the observed
    counts are then taken from ``data.deaths`` and ``observed`` is not needed.
    """
    var = _pick_obs_var(idata, series)
    obs = observed
    if _is_multiregion(idata, var) and obs is None:
        obs = True  # taken from data.deaths inside _series_plot
    kind = "col" if _is_multiregion(idata, var) else "point"
    return _series_plot(idata, var, data, group, levels, x, xlab, ylab,
                        _BLUES, None, save, "obs", title=title, obs=obs, obs_kind=kind)


# --------------------------------------------------------------------------
# Effect-size plots (multilevel)
# --------------------------------------------------------------------------


def _forest(df, xlab, title, hline=0.0):
    return (
        ggplot(df, aes("term", "median"))
        + geom_hline(yintercept=hline, linetype="dotted", color="#555555")
        + geom_pointrange(aes(ymin="lo", ymax="hi"))
        + coord_flip()
        + labs(x="", y=xlab, title=title)
        + theme_epidemia()
    )


def plot_effects(idata, group=None, labels=None, levels=90, save=True, title=None):
    """Forest plot of the NPI effects on the logit-``R_t`` scale.

    ``group=None`` shows the **global** (fixed) effects :math:`\\beta_k` — the
    average effect across regions. ``group="Italy"`` shows that region's total
    effect :math:`\\beta_k + b^{(m)}_k`, which is what actually drives its
    :math:`R_t` and can differ from the global value under partial pooling.
    """
    lo_q, hi_q = (100 - levels) / 2, 100 - (100 - levels) / 2
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
    lo_q, hi_q = (100 - levels) / 2, 100 - (100 - levels) / 2
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


def _region_path_frame(idata, var, data, idx, group=None):
    """Per-region path + median frames on real dates, padded days dropped.

    The same draw indices are used in every region: a draw is a joint sample over
    all regions, so pairing region A's draw 3 with region B's draw 7 would show a
    combination the posterior never produced.
    """
    arr, dims = _draws(idata, var)  # (draws, region, time)
    if dims[:1] != ("region",):
        raise ValueError(f"expected a region dim on {var!r}, got dims {dims}")
    regions = [str(r) for r in idata.posterior.coords["region"].values]
    keep = regions if group is None else [str(group)]
    unknown = set(keep) - set(regions)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; have {regions}")

    paths, meds = [], []
    for r in keep:
        m = regions.index(r)
        n = int(data.lengths[data.regions.index(r)])
        x = pd.to_datetime(data.dates[data.regions.index(r)])[:n]
        d = arr[:, m, :n]
        pf = _path_frame(d, x, idx)
        pf["region"] = r
        paths.append(pf)
        md = _median_frame(d, x)  # median over ALL draws, as in R
        md["region"] = r
        meds.append(md)
    return pd.concat(paths, ignore_index=True), pd.concat(meds, ignore_index=True), keep


def _spaghetti_plot(paths, med, color, alpha, ylab, xlab, hline=None, facet=0,
                    title=None, obs=None, obs_kind="point"):
    """``facet`` is the number of region panels (0/1 => no faceting)."""
    p = (
        ggplot()
        + geom_line(paths, aes("x", "value", group="draw"), color=color,
                    alpha=alpha, size=0.35)
    )
    if obs is not None:
        if obs_kind == "col":
            p = p + geom_col(obs, aes("x", "obs"), fill="#b2182b", alpha=0.45, width=1.0)
        else:
            p = p + geom_point(obs, aes("x", "obs"), color="#b2182b", size=1.1, alpha=0.8)
    # The median goes on last so it stays readable through the bundle of paths.
    p = p + geom_line(med, aes("x", "median"), color="black", size=0.8) + labs(x=xlab, y=ylab)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dotted", color="#555555")
    if title:
        p = p + labs(title=title)
    p = p + theme_epidemia()
    if facet:
        p = p + facet_wrap("region", scales="free_y", ncol=_FACET_NCOL)
        p = _size_for_panels(p, facet)
    return p


def _spaghetti_series(idata, var, data, group, draws, alpha, seed, x, xlab, ylab,
                      color, hline, save, default_name, title=None, obs=None,
                      obs_kind="point"):
    """Dispatch: multi-region (facet on real dates) vs single-population."""
    _check_alpha(alpha)
    arr, _ = _draws(idata, var)
    idx = _sample_draw_index(arr.shape[0], draws, seed)

    if _is_multiregion(idata, var):
        if data is None:
            raise ValueError(
                f"{var!r} has a 'region' dimension (a multilevel fit), so the "
                "MultilevelData you fitted is needed to map each region's columns "
                "back to its own dates. Pass data=<your prepare_panel result> "
                "(and optionally group='Italy' for a single region)."
            )
        paths, med, keep = _region_path_frame(idata, var, data, idx, group)
        obs_df = None
        if obs is True:  # sentinel: take the counts from the panel
            obs_df = _observed_frame(data, group)
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


def spaghetti_rt(idata, data=None, group=None, draws=50, alpha=0.3, seed=0,
                 x=None, xlab=None, save=True, title=None, region=None):
    """Reproduction numbers: ``draws`` individual posterior paths, median on top.

    The counterpart of :func:`plot_rt` for looking at trajectories rather than
    pointwise quantiles. ``draws`` caps the number of paths (fewer are drawn if
    the posterior is smaller) and ``seed`` fixes which ones, so the figure is
    reproducible. ``alpha`` is the per-path transparency: lower it when the
    bundle is dense.

    ``region`` is an alias for ``group``, kept for symmetry with R's ``groups``.
    """
    return _spaghetti_series(idata, "Rt", data, group if group is not None else region,
                             draws, alpha, seed, x, xlab, "$R_t$", _RT_PATH_COLOR,
                             1.0, save, "spaghetti-rt", title=title)


def spaghetti_infections(idata, data=None, group=None, draws=50, alpha=0.3, seed=0,
                         x=None, xlab=None, save=True, title=None, region=None):
    """Latent daily infections: ``draws`` individual posterior paths, median on top."""
    return _spaghetti_series(idata, "infections", data,
                             group if group is not None else region,
                             draws, alpha, seed, x, xlab, "Infections",
                             _OBS_PATH_COLOR, None, save, "spaghetti-infections",
                             title=title)


def spaghetti_obs(idata, observed=None, data=None, group=None, draws=50, alpha=0.3,
                  series=None,
                  seed=0, x=None, xlab=None, ylab="Daily deaths", save=True,
                  title=None, region=None):
    """Expected observations as individual paths, with the observed counts overlaid.

    Reads ``E_obs`` (single-population) or ``E_deaths`` (multilevel), whichever
    the fitted model defines; raises :class:`KeyError` if neither is present. For
    a multilevel fit pass ``data`` and the counts are taken from ``data.deaths``.
    """
    var = _pick_obs_var(idata, series)
    grp = group if group is not None else region
    obs = observed
    if _is_multiregion(idata, var) and obs is None:
        obs = True  # taken from data.deaths inside _spaghetti_series
    kind = "col" if _is_multiregion(idata, var) else "point"
    return _spaghetti_series(idata, var, data, grp, draws, alpha, seed, x, xlab, ylab,
                             _OBS_PATH_COLOR, None, save, "spaghetti-obs", title=title,
                             obs=obs, obs_kind=kind)
