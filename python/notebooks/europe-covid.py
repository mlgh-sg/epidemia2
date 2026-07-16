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
from plotnine import aes, geom_col, geom_line, geom_ribbon, geom_vline, ggplot, labs

import epidemia as epi
from epidemia.plots import save_plot, theme_epidemia

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
fit = epi.prepare_panel(
    data, epi.EUROPE_COVID_NPIS,
    seed_offset=30, death_threshold=10, fit_until="2020-05-05",
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
config = epi.MultilevelConfig(
    gen=ec.si,            # generation kernel (serial interval)
    i2o=ec.inf2death,     # infection-to-death delay
    R_link_K=6.5,         # scaled_logit(6.5) on R
    ifr_link_K=0.02,      # IFR in (0, 2%)
    seed_days=6,
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
idata = epi.fit_multilevel(
    fit, config, draws=1000, tune=2000, chains=4, seed=12345,
    adaptation="low_rank", target_accept=0.99,
)
print("divergences:", int(idata.sample_stats["diverging"].sum()))
az.summary(idata, var_names=["beta", "sd", "ifr", "reciprocal_dispersion", "seed_tau"])

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
epi.plots.plot_obs(idata, data=fit, save="deaths-ppc",
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

# %% [markdown]
# ## Forecasting and counterfactuals
#
# Forecasting in `epidemia` means swapping in a new data frame — extending the
# dates, or altering covariates for a *counterfactual*. We reproduce this here by
# forward-simulating the renewal process from the posterior draws, reusing the
# package's NumPy renewal reference (`epi.renewal_infections`). The function
# below takes a per-country NPI design (of any length) and returns posterior
# deaths, so the same code serves both out-of-sample forecasts and
# counterfactuals.


# %%
def posterior_deaths(idata, X_list, config, n_draws=400, seed=0):
    """Posterior expected deaths per region for arbitrary NPI designs.

    ``X_list[m]`` is a ``(T_m, K)`` design matrix (possibly longer than the
    fitted window, or with shifted policies). Returns a list of ``(n_draws,
    T_m)`` arrays of expected daily deaths.
    """
    post = idata.posterior
    rng = np.random.default_rng(seed)
    S = post.sizes["chain"] * post.sizes["draw"]
    take = rng.choice(S, size=min(n_draws, S), replace=False)

    def flat(name):
        a = np.asarray(post[name].stack(s=("chain", "draw")))
        return np.moveaxis(a, -1, 0)[take]           # (draws, ...)

    beta = flat("beta")                               # (D, K)
    b0 = flat("b0")                                   # (D, M)
    b = flat("b")                                     # (D, M, K)
    seed_ = flat("seed")                              # (D, M)
    ifr = flat("ifr")                                 # (D,)
    gen = np.asarray(config.gen)
    i2o = np.asarray(config.i2o)
    v = config.seed_days
    out = []
    for m, X in enumerate(X_list):
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        deaths = np.empty((len(take), T))
        for j in range(len(take)):
            eta = b0[j, m] + X @ (beta[j] + b[j, m])
            R = config.R_link_K / (1.0 + np.exp(-eta))
            seeds = np.full(v, seed_[j, m])
            infections = epi.renewal_infections(R, seeds, gen)
            # Use the package's own reference rather than a hand-rolled
            # convolution, so the forecast applies the *same* lag convention as
            # the model that produced these draws.
            deaths[j] = epi.expected_observations(infections, i2o, ifr[j])[:T]
        out.append(deaths)
    return out


# %% [markdown]
# **Out-of-sample forecast.** We rebuild the design for each country over the
# *full* period (through the end of the data), fit only to data before 5 May, and
# forecast beyond it. Here we show the United Kingdom.

# %%
full = epi.prepare_panel(data, epi.EUROPE_COVID_NPIS,
                         seed_offset=30, death_threshold=10, fit_until=None)
uk = full.regions.index("United_Kingdom")
X_uk = full.X[uk, : full.lengths[uk], :]
uk_dates = pd.to_datetime(full.dates[uk])
fc = posterior_deaths(idata, [X_uk], config, seed=1)[0]

uk_df = pd.DataFrame({
    "date": uk_dates,
    "median": np.median(fc, axis=0),
    "lo": np.percentile(fc, 2.5, axis=0),
    "hi": np.percentile(fc, 97.5, axis=0),
})
uk_obs = pd.DataFrame({
    "date": uk_dates,
    "deaths": full.deaths[uk, : full.lengths[uk]],
})
p = (
    ggplot(uk_df, aes("date"))
    + geom_col(uk_obs, aes("date", "deaths"), fill="#b2182b", alpha=0.5)
    + geom_ribbon(aes(ymin="lo", ymax="hi"), fill="#6baed6", alpha=0.5)
    + geom_line(aes(y="median"), color="black", size=0.5)
    + geom_vline(xintercept=pd.Timestamp("2020-05-05"), linetype="dotted",
                 color="#555555")
    + labs(x="", y="Daily deaths",
           title="United Kingdom: out-of-sample forecast (fit ends 5 May, dotted)")
    + theme_epidemia()
)
save_plot(p, "uk-forecast")
p

# %% [markdown]
# **Counterfactual: all policies 3 days earlier.** We shift each NPI indicator
# back three days for the UK and re-simulate deaths from the same posterior.

# %%
def shift_earlier(col, k):
    return np.concatenate([col[k:], np.ones(k)])


X_cf = X_uk.copy()
for j in range(X_cf.shape[1]):
    X_cf[:, j] = shift_earlier(X_cf[:, j], 3)
cf = posterior_deaths(idata, [X_cf], config, seed=1)[0]

cmp = pd.concat([
    pd.DataFrame({"date": uk_dates, "median": np.median(fc, axis=0),
                  "lo": np.percentile(fc, 2.5, axis=0),
                  "hi": np.percentile(fc, 97.5, axis=0), "scenario": "actual"}),
    pd.DataFrame({"date": uk_dates, "median": np.median(cf, axis=0),
                  "lo": np.percentile(cf, 2.5, axis=0),
                  "hi": np.percentile(cf, 97.5, axis=0), "scenario": "3 days earlier"}),
], ignore_index=True)
p = (
    ggplot(cmp, aes("date", color="scenario", fill="scenario"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.25, color=None)
    + geom_line(aes(y="median"), size=0.6)
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
