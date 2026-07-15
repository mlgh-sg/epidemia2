"""Plotting with plotnine (a grammar of graphics for Python).

These helpers mirror the R package's plots — credible-interval ribbons in a
single-hue sequential palette (wider interval = lighter) with the posterior
median on top — and ship a clean, publication-oriented theme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_minimal,
)

# single-hue sequential ramps (light -> dark), colour-blind friendly
_GREENS = {30: "#238b45", 60: "#74c476", 90: "#c7e9c0"}
_BLUES = {30: "#2171b5", 60: "#6baed6", 90: "#c6dbef"}


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


def _flatten(idata, var):
    """Posterior draws of ``var`` as a (draws, N) array (chains merged)."""
    da = idata.posterior[var]
    arr = np.asarray(da)  # (chain, draw, N)
    return arr.reshape(-1, arr.shape[-1])


def _interval_frame(draws, x, levels):
    """Long dataframe of central credible intervals at each ``level`` (percent)."""
    rows = []
    for lv in sorted(levels, reverse=True):  # widest first so it draws underneath
        lo = np.percentile(draws, (100 - lv) / 2, axis=0)
        hi = np.percentile(draws, 100 - (100 - lv) / 2, axis=0)
        rows.append(pd.DataFrame({"x": x, "lower": lo, "upper": hi, "level": str(lv)}))
    df = pd.concat(rows, ignore_index=True)
    df["level"] = pd.Categorical(df["level"], categories=[str(l) for l in sorted(levels)])
    return df


def _median_frame(draws, x):
    return pd.DataFrame({"x": x, "median": np.median(draws, axis=0)})


def _ribbon_plot(draws, x, palette, levels, ylab, xlab, hline=None):
    band = _interval_frame(draws, x, levels)
    med = _median_frame(draws, x)
    cols = {str(lv): palette[lv] for lv in levels}
    p = (
        ggplot()
        + geom_ribbon(band, aes("x", ymin="lower", ymax="upper", fill="level"))
        + scale_fill_manual(values=cols, name="Credible interval (%)")
        + geom_line(med, aes("x", "median"), color="black", size=0.7)
        + labs(x=xlab, y=ylab)
        + theme_epidemia()
    )
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dotted", color="#555555")
    return p


def plot_rt(idata, levels=(30, 60, 90), x=None, xlab="Day"):
    """Plot posterior credible intervals and median of the reproduction number."""
    draws = _flatten(idata, "Rt")
    if x is None:
        x = np.arange(draws.shape[1])
    return _ribbon_plot(draws, x, _GREENS, levels, ylab="$R_t$", xlab=xlab, hline=1.0)


def plot_infections(idata, levels=(30, 60, 90), x=None, xlab="Day"):
    """Plot posterior credible intervals and median of latent daily infections."""
    draws = _flatten(idata, "infections")
    if x is None:
        x = np.arange(draws.shape[1])
    return _ribbon_plot(draws, x, _BLUES, levels, ylab="Infections", xlab=xlab)


def plot_obs(idata, observed=None, levels=(30, 60, 90), x=None, xlab="Day", ylab="Observations"):
    """Plot expected-observation credible intervals, with observed data overlaid."""
    draws = _flatten(idata, "E_obs")
    if x is None:
        x = np.arange(draws.shape[1])
    p = _ribbon_plot(draws, x, _BLUES, levels, ylab=ylab, xlab=xlab)
    if observed is not None:
        obs = np.asarray(observed, dtype=float)
        odf = pd.DataFrame({"x": x, "obs": obs})
        odf = odf[np.isfinite(odf["obs"])]
        p = p + geom_point(odf, aes("x", "obs"), color="#b2182b", size=1.1, alpha=0.8)
    return p
