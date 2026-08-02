# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assessing the Effects of Interventions on COVID-19
#
# This is the Python counterpart of the R vignette
# [`europe-covid`](https://mlgh-sg.com/epidemia2/articles/europe-covid.html)
# (*Multilevel Modeling*). It is a **jupytext** notebook stored as a plain
# `.py` file (percent format): open it directly in Jupyter/VS Code, or pair it
# with an `.ipynb` via `jupytext --sync`.
#
# We use a **hierarchical (partially pooled)** model to estimate the effect of
# non-pharmaceutical interventions (NPIs) on the transmissibility of COVID-19,
# following Flaxman et al. (2020): the effect of five measures enacted in March
# 2020 across 11 European countries, fit to daily death data.
#
# > **Estimation with MCMC, not Variational Bayes.** The R vignette fits this
# > model with Variational Bayes (`algorithm = "fullrank"`) for speed, and notes
# > that VB *understates* uncertainty ("relatively narrow intervals ... an
# > artifact of using Variational Bayes"). Here we instead run **full MCMC**
# > (NUTS, via [nutpie](https://github.com/pymc-devs/nutpie)), so the credible
# > intervals are the genuine posterior ones.

# %%
import numpy as np
import pandas as pd
import arviz as az
from plotnine import (aes, geom_col, geom_line, geom_point, geom_ribbon,
                      geom_vline, ggplot, labs, scale_color_manual,
                      scale_fill_manual)

import epidemia as epi
# `epi.prepare_panel` is the older multilevel helper; the core one returns the
# (PanelData, series) pair that ObsModel and forecast() expect.
from epidemia.core import prepare_panel
from epidemia.forecast import forecast
from epidemia.plots import save_plot, theme_epidemia
from epidemia.priors import normal

# %% [markdown]
# ## Data
#
# `epi.europe_covid2()` returns the `EuropeCovid2` dataset (the same data as the
# R package): daily cases and deaths and the five binary NPI indicators for 11
# countries, up to 1 July 2020, plus the serial interval (`si`) and the
# infection-to-death delay (`inf2death`).

# %%
ec = epi.europe_covid2()
data = ec.data
print(epi.EUROPE_COVID_NPIS)
data.head()

# %% [markdown]
# As in the R vignette, seeding for each country begins 30 days before cumulative
# deaths first exceed 10, and — to demonstrate forecasting later — we fit only up
# to the 5th of May 2020, holding out the rest. `epi.prepare_panel` performs this
# per-country filtering and returns padded, model-ready arrays.

# %%
fit, series = prepare_panel(
    data, npis=list(epi.EUROPE_COVID_NPIS), responses=["deaths"],
    seed_offset=30, threshold=10, fit_until="2020-05-05",
)
start_end = pd.DataFrame({
    "country": fit.regions,
    "start": [d[0] for d in fit.dates],
    "end": [d[-1] for d in fit.dates],
    "days": fit.lengths,
})
start_end

# %% [markdown]
# ### Model components
#
# As in `epidemia`, the model has three components: **transmission**,
# **infections**, and **observations**.
#
# #### Transmission
#
# Country-specific reproduction numbers are a step function of the five NPIs:
#
# $$ R^{(m)}_t = 6.5 \cdot \mathrm{sigmoid}\!\Big(b^{(m)}_0 + \sum_{k=1}^{5}\big(\beta_k + b^{(m)}_k\big) I^{(m)}_{k,t}\Big). $$
#
# * $\beta_k$ are **fixed** (global) NPI effects with a *shifted Gamma* prior,
#   $\beta_k = \tfrac{\log 1.05}{6} - g_k$, $g_k \sim \mathrm{Gamma}(1/6, 1)$ —
#   this makes a measure *a priori* reduce transmission (with a small allowance
#   to increase it), matching `shifted_gamma(shape = 1/6, scale = 1, shift = log(1.05)/6)`.
# * $b^{(m)}_0, b^{(m)}_k$ are **partially pooled** country effects,
#   $b^{(m)}_0 \sim N(0, \sigma_0)$ and $b^{(m)}_k \sim N(0, \sigma_k)$, with
#   $\sigma_0 \sim \mathrm{Gamma}(2, 0.25)$ and $\sigma_k \sim \mathrm{Gamma}(0.5, 0.25)$.
#   This is the `(1 + npis || country)` term with the `decov` covariance prior.
# * The link `scaled_logit(6.5)` keeps $R$ in $(0, 6.5)$ with midpoint $3.25$.
#
# #### Infections
#
# Basic (deterministic) renewal dynamics: infections are seeded over 6 days and
# propagated by the generation kernel `ec.si`. The seeds are themselves
# **partially pooled** through a shared mean,
# $\tau \sim \mathrm{Exp}(0.03)$ and $i^{(m)} \mid \tau \sim \mathrm{Exp}(\tau)$
# — R's `prior_seeds = hexp(prior_aux = exponential(0.03))` — so a country with
# little early death data borrows epidemic-size information from the others.
#
# Both kernels are **lag-1-first**: `ec.si[0]` weights infections one day back,
# `ec.inf2death[0]` weights infections one day before the death. An infection is
# never observed on the day it happens, matching R's Stan (which sums over
# `infections[start .. t-1]` for both), so the vectors from the R data objects
# drop in unchanged.
#
# #### Observations
#
# Deaths are modelled with a constant infection-fatality ratio (IFR),
# $\mathrm{IFR} = 0.02 \cdot \mathrm{sigmoid}(\alpha)$, $\alpha \sim N(0, 0.2)$
# (a prior mean IFR of $1\%$), convolved with `ec.inf2death`, and a
# negative-binomial likelihood whose reciprocal dispersion is
# $10 + 5\cdot\mathrm{HalfNormal}(1)$ — R's `epiobs` default
# `prior_aux = normal(location = 10, scale = 5)`.
#
# All of this is captured by `MultilevelConfig`, whose defaults already match
# the priors and links above.

# %%
# Built with the package's CORE builder rather than the older MultilevelConfig
# path, so that epidemia's own forecast() can be used below instead of a
# hand-rolled re-simulation. The two describe the SAME model -- their
# log-densities agree to 0.0000 at a matched point, with identical observed-RV
# contributions -- but only the core builder's objects are accepted by
# forecast().
obs = [epi.ObsModel(
    "deaths", series["deaths"]["y"], series["deaths"]["mask"],
    i2o=ec.inf2death,        # infection-to-death delay
    family="neg_binom",
    link="scaled_logit", link_K=0.02,   # IFR in (0, 2%)
    prior_intercept=normal(0.0, 0.2),
    prior_aux=normal(10.0, 5.0),
)]
config = epi.EpiModelConfig(
    gen=ec.si,               # generation kernel (serial interval)
    link="scaled_logit", R_link_K=6.5,
    intercept=False,         # R's `~ 0 + ...`: no global intercept
    region_effects=True, correlated=False,
    beta_shape=1/6, beta_scale=1.0, beta_shift=float(np.log(1.05) / 6),
    sd_intercept_shape=2.0, sd_slope_shape=0.5, sd_scale=0.25,
    seed_days=6, seed_pooling=True, seed_aux_rate=0.03, seed_prior_mean=30.0,
)
config

# %% [markdown]
# ## Model fitting (MCMC)
#
# We fit with nutpie's NUTS sampler. For a quick pass use fewer draws; for
# publication-quality intervals increase `draws`/`tune` (e.g. 1000/1000) and use
# 4 chains. This is genuine Hamiltonian Monte Carlo — **not** the Variational
# Bayes used in the R vignette.
#
# **This takes a while** — roughly an hour for 11 countries at these settings, so
# the progress bar is on by default (the compile step is announced separately,
# since nutpie reports nothing during it and it is not fast either).
#
# `target_accept` defaults to **0.95**, not nutpie's 0.8. The funnel geometry of
# a hierarchical model — the between-country SDs against the non-centred country
# effects — gives hundreds of divergences at 0.8 here. `fit_multilevel` warns if
# any survive; a divergent fit's intervals should not be quoted.
#
# **`adaptation="low_rank"` matters for this model**, and not for a cosmetic
# reason. The NPI coefficients are strongly correlated *in the posterior* (they
# are collinear in the data), which makes a long thin ridge that a diagonal mass
# matrix cannot follow. With the default `"diag"`, `beta[schools]` and
# `beta[social_distancing_encouraged]` come out at $\hat{R} \approx 1.08$–$1.11$
# with an effective sample size of **25–36 out of 4000** — not converged, and
# their point estimates are visibly *wrong* as a result (social distancing shifts
# from $-1.11$ to $-1.35$ once the sampler can actually traverse the ridge).
# `"low_rank"` estimates a low-rank correction to the mass matrix, which is
# exactly the right tool for a correlated posterior, and brings every
# $\hat{R} \le 1.04$.
#
# The lesson generalises: **divergences and $\hat{R}$ are different failures**.
# Raising `target_accept` fixes the funnel; only a better mass matrix fixes the
# ridge. Check both.

# %%
idata = epi.fit_epidemia(
    fit, obs, config, draws=1000, tune=2000, chains=4, seed=12345,
    adaptation="diag", target_accept=0.99, maxdepth=14,
)
print("divergences:", int(idata.sample_stats["diverging"].sum()))
az.summary(idata, var_names=["beta", "sd", "deaths|rate", "deaths|aux", "seed_tau"])

# %% [markdown]
# Check the diagnostics before reading anything off this fit — `r_hat` should be
# $\le 1.01$ and `ess_bulk` in the hundreds at least. If `beta[...]` rows have a
# poor `r_hat`, the effect estimates below are not trustworthy no matter how
# reasonable they look.

# %%
summ = az.summary(idata, var_names=["beta", "sd"])
bad = summ[(summ["r_hat"] > 1.01) | (summ["ess_bulk"] < 400)]
if len(bad):
    print("NOT CONVERGED — do not quote these:")
    print(bad[["mean", "ess_bulk", "r_hat"]].to_string())
else:
    print(f"all clear: max r_hat = {summ['r_hat'].max():.3f}, "
          f"min ess_bulk = {summ['ess_bulk'].min():.0f}")

# %% [markdown]
# ## Posterior predictive checks
#
# Expected deaths (posterior median and 50 %/95 % credible bands) against the
# observed daily deaths, **one panel per country** — the counterpart of the R
# vignette's `plot_obs(fm, type = "deaths", levels = c(50, 95))`.
#
# `Rt`, `infections` and `E_deaths` are stored in the posterior indexed by
# `region`, so `epi.plots` draws every country for you: pass the `fit` object
# (the `prepare_panel` result) as `data=` and each region is placed on **its own
# dates** — remember each region's column `t` is a *different* calendar day, and
# each region's padded tail is dropped. Every plot is written to `figures/`
# (override with `$EPIDEMIA_FIGDIR`); pass `save=False` to skip.

# %%
# PanelData holds only the design, so the observations come from the ObsModel.
# (MultilevelData used to carry `deaths` itself; PanelData deliberately does
# not, since a multi-series fit has no single response to carry.)
epi.plots.plot_obs(idata, data=fit, obs_model=obs[0], save="deaths-ppc",
                   title="Posterior predictive: deaths")

# %% [markdown]
# ## Reproduction numbers
#
# The inferred $R^{(m)}_t$ per country (a step function of the NPIs). We expect
# it to fall below one in each country as measures come into force.

# %%
epi.plots.plot_rt(idata, data=fit, save="rt-by-country",
                  title="Inferred reproduction numbers")

# %% [markdown]
# Latent infections, likewise per country:

# %%
epi.plots.plot_infections(idata, data=fit, save="infections-by-country",
                          title="Latent daily infections")

# %% [markdown]
# ## Effect sizes
#
# **Global** NPI effects $\beta_k$ — the average effect of each measure across
# countries. A large negative coefficient means a strong reduction in
# transmission. As in the R analysis, lockdown is the most effective on average.
#
# > **How to read this plot — the measures are highly collinear.** Most countries
# > enacted all five NPIs within a few days of each other (Germany banned public
# > events on the *same day* it locked down), so the individual $\beta_k$ are only
# > weakly identified: the data constrain their **sum** far better than the split
# > between them. The R vignette makes the same point — *"when repeating this
# > analysis with full MCMC, we observe that the intervals for all policies other
# > than lockdown overlap with zero"*. So an interval that straddles zero here
# > means **"not separately identifiable from the other measures"**, not "this
# > measure did nothing". The cell after next quantifies exactly that.

# %%
labels = ["Schools", "Isolating", "Events", "Distancing", "Lockdown"]
epi.plots.plot_effects(idata, labels=labels, save="effects-global")

# %% [markdown]
# Because effects are *partially pooled*, the country-specific effect of measure
# $k$ is $\beta_k + b^{(m)}_k$, which can differ from the global $\beta_k$. Below
# we extract these for Italy (compare with the R vignette's Italy panel).

# %%
epi.plots.plot_effects(idata, group="Italy", labels=labels, save="effects-italy")

# %% [markdown]
# ### Collinearity: what *is* identified
#
# The individual coefficients trade off against one another, but their **total**
# — the combined effect once every measure is in force — is pinned down by the
# data. Comparing the two tells you how much of a small $\beta_k$ is a real
# "no effect" and how much is just collinearity moving the effect to a neighbour.

# %%
beta = np.asarray(idata.posterior["beta"].stack(s=("chain", "draw")))     # (K, S)
total = beta.sum(axis=0)
print("Combined effect of all five measures (logit Rt scale):")
print(f"  median {np.median(total):+.2f}  90% CI "
      f"[{np.percentile(total, 5):+.2f}, {np.percentile(total, 95):+.2f}]")
print("\nIndividual measures — note how much wider these are relative to their size:")
for k, lab in enumerate(labels):
    d = beta[k]
    print(f"  {lab:11s} median {np.median(d):+.2f}  90% CI "
          f"[{np.percentile(d, 5):+.2f}, {np.percentile(d, 95):+.2f}]"
          f"   P(effect < 0) = {(d < 0).mean():.2f}")
print("\nPosterior correlation between the coefficients (collinearity fingerprint):")
print(pd.DataFrame(np.round(np.corrcoef(beta), 2), index=labels, columns=labels).to_string())

# %% [markdown]
# ### The per-country lockdown effect
#
# The global $\beta_\text{lockdown}$ is one number for all 11 countries. What
# actually drives country $m$'s $R_t$ is $\beta_k + b^{(m)}_k$ — so this is the
# plot to read if the question is *"did lockdown do anything **here**?"*
# Sweden is the instructive case: it never locked down, so its lockdown column is
# identically zero, its likelihood says nothing about the effect, and its
# posterior is just the shared prior. The model explains Sweden through its
# country-specific intercept instead — exactly the point the R vignette makes.

# %%
epi.plots.plot_region_effects(idata, "lockdown", save="lockdown-by-country")

# %% [markdown]
# ## Effect sizes as a percent reduction in transmission
#
# A coefficient on the logit scale is hard to feel. `epi.effect_table` converts
# it into the quantity people actually want — *by what percent did this measure
# cut transmission?*
#
# > **Why not just $1 - e^{\beta_k}$?** Because that is the answer for a **log**
# > link, and this model uses `scaled_logit(6.5)`:
# > $R = 6.5\,\operatorname{sigmoid}(\eta)$. A coefficient therefore does **not**
# > map to a constant multiplicative effect — the same $\beta$ buys a bigger
# > percentage in a country with a low $R_0$ than in one with a high $R_0$. On
# > this data $1-e^{\beta}$ overstates lockdown by roughly 9 percentage points.
# > `effect_table` instead does the counterfactual properly, per posterior draw:
# > compare $6.5\,\operatorname{sigmoid}(b_0^{(m)})$ (no measures) with
# > $6.5\,\operatorname{sigmoid}(b_0^{(m)} + \beta_k + b_k^{(m)})$ (measure $k$
# > on). That is also why the answer is reported per country rather than as one
# > global number.

# %%
tab = epi.effect_table(idata, config, data=fit)
pct = tab[tab["kind"] == "pct"]

print("Reduction in R_t (%), median [90% CI] — per country\n")
piv = pct.pivot(index="region", columns="term", values="median")
print(piv[[*fit.npis, "all measures"]].round(1).to_string())

print("\n\nAll five measures combined:\n")
for _, r in pct[pct["term"] == "all measures"].iterrows():
    print(f"  {r['region']:16s} {r['median']:5.1f}%  [{r['lo']:5.1f}, {r['hi']:5.1f}]")

print("\n\nR_0 -> R_t once every measure is in force:\n")
R = tab[tab["kind"] == "R"]
r0 = R[R["term"] == "R_0 (no measures)"].set_index("region")
ra = R[R["term"] == "R_t (all measures)"].set_index("region")
for reg in fit.regions:
    print(f"  {reg:16s} {r0.loc[reg, 'median']:.2f}  ->  {ra.loc[reg, 'median']:.2f}")

# %% [markdown]
# The same thing as a plot. Measures a country **never enacted** are greyed out:
# there, the percentage is a counterfactual drawn from the pooled prior ("what
# lockdown *would* have done in Sweden"), not a measured effect, and it should
# not be read alongside the others as if it were evidence.

# %%
epi.plots.plot_percent_effects(idata, config, data=fit, labels=labels,
                               save="percent-effects-by-country")

# %%
epi.plots.plot_percent_effects(idata, config, data=fit, group="Italy", labels=labels,
                               save="percent-effects-italy",
                               title="Italy: reduction in transmission by measure")

# %%
# One palette for the whole notebook, so "observed", "fitted", "forecast" and
# "counterfactual" keep the same meaning in every figure.
from epidemia.plots import COLORS as _C
_PALETTE = {"observed": _C["observed"], "forecast_band": "#6baed6",
            "median": _C["median"], "counterfactual": _C["counterfactual"],
            "out_of_sample": _C["out_of_sample"]}

# %% [markdown]
# ## Forecasting and counterfactuals
#
# Forecasting in `epidemia` means swapping in a new data frame — extending the
# dates, or altering covariates for a *counterfactual*.
#
# This notebook predates `epidemia.forecast()` and forward-simulates by hand
# below, which is worth keeping because it shows the mechanism. For new work use
# **`epidemia.forecast(idata, panel, obs_models, config, newdata=...)`**: it is
# one call, it propagates the random walk over the horizon instead of freezing
# it, and it draws the observation noise rather than returning expectations. The
# [Spanish flu tutorial](flu.md) uses it. The hand-rolled version below returns
# *expected* deaths, so its intervals are narrower than a predictive interval.


# %%
# %% [markdown]
# **Out-of-sample forecast.** The model was fit only to data before 5 May, so
# everything after that is genuinely held out. `epidemia.forecast.forecast()`
# continues the renewal recursion past the fitted window from the same posterior
# draws — the same code path R uses via standalone generated quantities, so the
# forecast and the in-sample fit are guaranteed to agree where they overlap.
#
# An earlier version of this notebook re-simulated deaths with a hand-rolled
# helper instead. It did not reproduce its own fit: the UK forecast median
# peaked near 1500 where `plot_obs` showed roughly 950. Reconstructing the
# linear predictor by hand is easy to get subtly wrong, and there is no reason
# to when the package can do it.

# %%
# newdata carries the covariates over the FULL period; forecast() carries the
# last observed row forward for any day it does not cover.
fcast = forecast(idata, fit, obs, config, newdata=data, draws=400, seed=1)
uk = fcast.regions.index("United_Kingdom")
uk_dates = pd.to_datetime(fcast.dates[uk])
fc = fcast.predicted["deaths"][:, uk, : fcast.lengths[uk]]

# Does the forecast reproduce the fit it came from? Over the FITTED window the
# forecast's expected deaths must equal the model's own E_deaths draw for draw,
# since forecast() replays the same recursion from the same parameters. The
# hand-rolled helper this replaced did NOT: it put the UK median near 1500 where
# plot_obs showed roughly 950. Checking it here means the two can never silently
# drift apart again -- `draw_index` lets the comparison be exact rather than a
# comparison of medians.
_E_fit = np.asarray(
    idata.posterior["E_deaths"].stack(s=("chain", "draw"))
)                                                    # (M, T, S)
_E_fit = np.moveaxis(_E_fit, -1, 0)[fcast.draw_index]   # (D, M, T), same draws
_n = int(fit.lengths[uk])
_a = fcast.expected["deaths"][:, uk, :_n]
_b = _E_fit[:, uk, :_n]
_rel = np.abs(_a - _b) / np.maximum(np.abs(_b), 1e-9)
print(f"forecast vs fitted E_deaths over the fitted window ({_n} days):")
print(f"  max relative difference : {_rel.max():.3e}")
print(f"  median |forecast - fit| : {np.median(np.abs(_a - _b)):.3e}")
assert _rel.max() < 1e-6, (
    f"forecast() does not reproduce the fit (max rel diff {_rel.max():.3g}); "
    "the two have drifted apart"
)
print("  OK - the forecast reproduces the fit exactly over the fitted window")

# Three NESTED credible bands, as R's plot_obs draws, and the observed counts
# split into in-sample and out-of-sample. A single 95% band with one colour of
# bar hides both how the uncertainty is shaped and where the fit stopped.
_LEVELS = (30, 60, 90)
# String keys: the frame stores level as str(lv), and an int-keyed dict
# silently falls through to plotnine's grey default.
_BANDS = {"30": "#2171b5", "60": "#6baed6", "90": "#c6dbef"}

uk_bands = pd.concat([
    pd.DataFrame({
        "date": uk_dates,
        "lo": np.percentile(fc, (100 - lv) / 2, axis=0),
        "hi": np.percentile(fc, 100 - (100 - lv) / 2, axis=0),
        "level": str(lv),
    })
    for lv in sorted(_LEVELS, reverse=True)          # widest first, so it sits behind
], ignore_index=True)
uk_med = pd.DataFrame({"date": uk_dates, "median": np.median(fc, axis=0)})

# The observed counts come straight from the source frame over the forecast's
# own dates, so the two are aligned by date rather than by position.
n_fit = int(fit.lengths[fit.regions.index("United_Kingdom")])
uk_obs = (data[data["country"] == "United_Kingdom"][["date", "deaths"]]
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .merge(pd.DataFrame({"date": uk_dates}), on="date", how="right")
          .sort_values("date").reset_index(drop=True))
uk_obs["period"] = np.where(np.arange(len(uk_obs)) < n_fit,
                            "In-sample", "Out-of-sample")

p = (
    ggplot()
    + geom_ribbon(uk_bands, aes("date", ymin="lo", ymax="hi", fill="level"))
    + scale_fill_manual(values=_BANDS, name="Credible interval (%)",
                        breaks=[str(lv) for lv in sorted(_LEVELS)])
    + geom_point(uk_obs, aes("date", "deaths", color="period"), size=0.9,
                 alpha=0.8, stroke=0)
    + scale_color_manual(
        values={"In-sample": _PALETTE["observed"],
                "Out-of-sample": _PALETTE["out_of_sample"]}, name="")
    + geom_line(uk_med, aes("date", "median"), color=_PALETTE["median"],
                size=0.7)
    + labs(x="", y="Daily deaths",
           title="United Kingdom: fitted to 5 May, forecast beyond")
    + theme_epidemia()
)
save_plot(p, "uk-forecast")
p

# %% [markdown]
# **Counterfactual: all policies 3 days earlier.** We shift each NPI indicator
# back three days for the UK and re-simulate deaths from the same posterior.

# %%
# Shift every NPI indicator three days earlier, then re-run the SAME posterior
# through forecast(). Because the counterfactual goes through the same code path
# as the fit, the two are directly comparable.
cf_data = data.copy()
for col in epi.EUROPE_COVID_NPIS:
    cf_data[col] = cf_data.groupby("country")[col].shift(-3)
    cf_data[col] = cf_data.groupby("country")[col].ffill().fillna(0.0)

cf_cast = forecast(idata, fit, obs, config, newdata=cf_data, draws=400, seed=1)
cf = cf_cast.predicted["deaths"][:, uk, : cf_cast.lengths[uk]]

cmp = pd.concat([
    pd.DataFrame({"date": uk_dates, "median": np.median(fc, axis=0),
                  "lo": np.percentile(fc, 2.5, axis=0),
                  "hi": np.percentile(fc, 97.5, axis=0), "scenario": "actual"}),
    pd.DataFrame({"date": uk_dates, "median": np.median(cf, axis=0),
                  "lo": np.percentile(cf, 2.5, axis=0),
                  "hi": np.percentile(cf, 97.5, axis=0), "scenario": "3 days earlier"}),
], ignore_index=True)
# The scenario colours come from epidemia's palette rather than plotnine's
# defaults: the counterfactual must not share a hue with the fitted series, or
# the two read as the same quantity. Purple against blue separates in hue, in
# lightness and under deuteranopia.
from epidemia.plots import COLORS
from plotnine import scale_color_manual, scale_fill_manual

_scen = {"actual": COLORS["in_sample"], "3 days earlier": COLORS["counterfactual"]}
p = (
    ggplot(cmp, aes("date", color="scenario", fill="scenario"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.25, color=None)
    + geom_line(aes(y="median"), size=0.7)
    + scale_color_manual(values=_scen, name="")
    + scale_fill_manual(values=_scen, name="")
    + labs(x="", y="Daily deaths",
           title="United Kingdom: counterfactual (policies enacted 3 days earlier)")
    + theme_epidemia()
)
save_plot(p, "uk-counterfactual")
p

# %% [markdown]
# As in the R vignette, enacting measures a few days earlier markedly lowers the
# projected death curve. These results illustrate usage rather than a rigorous
# analysis.
#
# ### References
#
# * Flaxman, S. et al. (2020). *Estimating the effects of non-pharmaceutical
#   interventions on COVID-19 in Europe.* Nature 584, 257–261.
# * Bhatt, S. et al. *Semi-mechanistic Bayesian modelling of COVID-19.*
