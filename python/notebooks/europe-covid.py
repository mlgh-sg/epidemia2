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
from plotnine import (
    aes, geom_col, geom_hline, geom_line, geom_ribbon, geom_pointrange,
    ggplot, facet_wrap, labs, coord_flip,
)

import epidemia as epi
from epidemia.plots import theme_epidemia

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
# propagated by the generation kernel `ec.si`.
#
# #### Observations
#
# Deaths are modelled with a constant infection-fatality ratio (IFR),
# $\mathrm{IFR} = 0.02 \cdot \mathrm{sigmoid}(\alpha)$, $\alpha \sim N(0, 0.2)$
# (a prior mean IFR of $1\%$), convolved with `ec.inf2death`, and a
# negative-binomial likelihood.
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

# %%
idata = epi.fit_multilevel(
    fit, config, draws=1000, tune=1000, chains=4, seed=12345, progress_bar=True,
)
print("divergences:", int(idata.sample_stats["diverging"].sum()))
az.summary(idata, var_names=["beta", "sd", "ifr", "reciprocal_dispersion"])

# %% [markdown]
# ## Posterior predictive checks
#
# Expected deaths (posterior median and 50 %/95 % credible bands) against the
# observed daily deaths, one panel per country. Because `Rt`, `infections` and
# `E_deaths` are stored in the posterior indexed by `region`, we can build the
# per-country plots directly.


# %%
def _region_bands(idata, var, regions, dates, lengths, levels=(50, 95)):
    """Long dataframe of per-region median + credible bands for a latent series."""
    da = idata.posterior[var]                       # (chain, draw, region, time)
    arr = np.asarray(da).reshape(-1, da.shape[-2], da.shape[-1])  # (draws, M, T)
    rows = []
    for m, r in enumerate(regions):
        n = int(lengths[m])
        x = pd.to_datetime(dates[m])
        d = arr[:, m, :n]
        row = {"date": x, "median": np.median(d, axis=0), "country": r}
        for lv in levels:
            row[f"lo{lv}"] = np.percentile(d, (100 - lv) / 2, axis=0)
            row[f"hi{lv}"] = np.percentile(d, 100 - (100 - lv) / 2, axis=0)
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def _observed_frame(deaths, regions, dates, lengths):
    rows = []
    for m, r in enumerate(regions):
        n = int(lengths[m])
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(dates[m]), "deaths": deaths[m, :n], "country": r,
        }))
    return pd.concat(rows, ignore_index=True)


bands = _region_bands(idata, "E_deaths", fit.regions, fit.dates, fit.lengths)
obs = _observed_frame(fit.deaths, fit.regions, fit.dates, fit.lengths)

(
    ggplot(bands, aes("date"))
    + geom_col(obs, aes("date", "deaths"), fill="#b2182b", alpha=0.5)
    + geom_ribbon(aes(ymin="lo95", ymax="hi95"), fill="#6baed6", alpha=0.5)
    + geom_ribbon(aes(ymin="lo50", ymax="hi50"), fill="#2171b5", alpha=0.6)
    + geom_line(aes(y="median"), color="black", size=0.5)
    + facet_wrap("country", scales="free_y")
    + labs(x="", y="Daily deaths", title="Posterior predictive: deaths")
    + theme_epidemia()
)

# %% [markdown]
# ## Reproduction numbers
#
# The inferred $R^{(m)}_t$ per country (a step function of the NPIs). We expect
# it to fall below one in each country as measures come into force.

# %%
rt_bands = _region_bands(idata, "Rt", fit.regions, fit.dates, fit.lengths)
(
    ggplot(rt_bands, aes("date"))
    + geom_ribbon(aes(ymin="lo95", ymax="hi95"), fill="#74c476", alpha=0.5)
    + geom_ribbon(aes(ymin="lo50", ymax="hi50"), fill="#238b45", alpha=0.6)
    + geom_line(aes(y="median"), color="black", size=0.5)
    + geom_hline(yintercept=1.0, linetype="dotted", color="#555555")
    + facet_wrap("country", scales="free_y")
    + labs(x="", y="$R_t$", title="Inferred reproduction numbers")
    + theme_epidemia()
)

# %% [markdown]
# ## Effect sizes
#
# **Global** NPI effects $\beta_k$ (the average effect across countries). A large
# negative coefficient means a strong reduction in transmission. As in the R
# analysis, lockdown is the most effective on average.

# %%
beta = idata.posterior["beta"].stack(sample=("chain", "draw")).transpose("sample", "npi")
beta = np.asarray(beta)                              # (draws, K)
labels = ["Schools", "Isolating", "Events", "Distancing", "Lockdown"]
eff = pd.DataFrame({
    "npi": labels,
    "median": np.median(beta, axis=0),
    "lo": np.percentile(beta, 5, axis=0),
    "hi": np.percentile(beta, 95, axis=0),
})
(
    ggplot(eff, aes("npi", "median"))
    + geom_hline(yintercept=0.0, linetype="dotted", color="#555555")
    + geom_pointrange(aes(ymin="lo", ymax="hi"))
    + coord_flip()
    + labs(x="", y="Global effect on logit $R_t$", title="Global NPI effects $\\beta_k$")
    + theme_epidemia()
)

# %% [markdown]
# Because effects are *partially pooled*, the country-specific effect of measure
# $k$ is $\beta_k + b^{(m)}_k$, which can differ from the global $\beta_k$. Below
# we extract these for Italy (compare with the R vignette's Italy panel).

# %%
b_italy = idata.posterior["b"].sel(region="Italy").stack(sample=("chain", "draw"))
b_italy = np.asarray(b_italy.transpose("sample", "npi"))   # (draws, K)
b0_italy = np.asarray(idata.posterior["b0"].sel(region="Italy").stack(sample=("chain", "draw")))
mat = np.column_stack([b0_italy, beta + b_italy])
cols = ["Intercept"] + labels
italy = pd.DataFrame({
    "term": cols,
    "median": np.median(mat, axis=0),
    "lo": np.percentile(mat, 5, axis=0),
    "hi": np.percentile(mat, 95, axis=0),
})
(
    ggplot(italy, aes("term", "median"))
    + geom_hline(yintercept=0.0, linetype="dotted", color="#555555")
    + geom_pointrange(aes(ymin="lo", ymax="hi"))
    + coord_flip()
    + labs(x="", y="Effect (logit $R_t$ scale)", title="Italy-specific effects $\\beta_k + b^{(Italy)}_k$")
    + theme_epidemia()
)

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
            conv = np.convolve(infections, i2o)[:T]
            deaths[j] = ifr[j] * conv
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
cutoff = pd.Timestamp("2020-05-05")
(
    ggplot(uk_df, aes("date"))
    + geom_col(uk_obs, aes("date", "deaths"), fill="#b2182b", alpha=0.5)
    + geom_ribbon(aes(ymin="lo", ymax="hi"), fill="#6baed6", alpha=0.5)
    + geom_line(aes(y="median"), color="black", size=0.5)
    + geom_hline(yintercept=0, color="white", size=0)  # keep y at 0
    + labs(x="", y="Daily deaths",
           title="United Kingdom: out-of-sample forecast (fit ends 5 May, dotted)")
    + theme_epidemia()
)

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
(
    ggplot(cmp, aes("date", color="scenario", fill="scenario"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.25, color=None)
    + geom_line(aes(y="median"), size=0.6)
    + labs(x="", y="Daily deaths",
           title="United Kingdom: counterfactual (policies enacted 3 days earlier)")
    + theme_epidemia()
)

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
