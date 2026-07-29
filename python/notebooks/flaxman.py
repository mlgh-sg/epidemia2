# WORK IN PROGRESS -- this notebook does NOT yet reproduce the Nature figure.
# See the "Where this does not reproduce" section at the end. It is deliberately
# excluded from ALL_TUTORIALS in scripts/precompute.py and from the mkdocs nav.

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
# # Reproducing Flaxman et al. (2020)
#
# The [Nature paper](https://www.nature.com/articles/s41586-020-2405-7) that
# **epidemia** grew out of reports one headline result: **lockdown accounts for
# essentially all of the measured reduction in transmission**, and the other four
# interventions sit near zero.
#
# The `epidemia` intervention vignette does *not* reproduce that. It puts a large
# effect on social distancing as well. That is not a bug in either package — the
# two are **different models**, and this notebook shows exactly how, then scores
# them against each other.
#
# ## Where the two specifications differ
#
# From `stan-models/base.stan` in the paper's repository:
#
# ```stan
# alpha_hier ~ gamma(.1667, 1);
# alpha[i]    = alpha_hier[i] - log(1.05) / 6.0;
# mu          ~ normal(3.28, kappa);     kappa ~ normal(0, 0.5);
# lockdown    ~ normal(0, gamma);
# Rt[,m]      = mu[m] * exp(-X[m] * alpha - X[m][,5] * lockdown[m]);
# tau ~ exponential(0.03);  y[m] ~ exponential(1/tau);
# ```
#
# | | Flaxman | epidemia vignette |
# |---|---|---|
# | Link | `mu[m] * exp(-X·alpha)` | `6.5 * sigmoid(eta)` |
# | Covariates | **six** — the five NPIs plus `firstIntervention` | the five NPIs |
# | Country deviations | **lockdown only** | **all five NPIs** |
# | Coefficient prior | `gamma(1/6, 1)`, shifted by `log(1.05)/6` | identical |
# | Seeding | `tau ~ Exp(0.03)`, `y ~ Exp(1/tau)` | identical |
# | Observations | negative binomial | identical |
#
# The priors, seeding and observation model are the *same*. Two things differ,
# and both matter.
#
# **The sixth covariate.** `firstIntervention` is 1 once *any* measure is in
# force. It absorbs the common "something changed" drop, leaving the five
# individual coefficients to explain only what distinguishes them from each
# other. Without it — as in the epidemia vignette — the five near-collinear
# columns must between them account for the entire fall in transmission, and
# which one takes the mass is largely arbitrary.
#
# **The pooling.** Only lockdown gets a country-specific term; the other five
# share one global coefficient each with no freedom to vary by country.
#
# Getting this wrong is instructive. A first draft of this notebook omitted
# `firstIntervention` and put a country deviation on lockdown alone — and
# returned lockdown at 0.4% and distancing at 63%, i.e. the *opposite* of the
# Nature figure. The sixth column is not a detail.

# %%
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm

import epidemia
from epidemia.core import (
    EpiModelConfig,
    ObsModel,
    PanelData,
    build_epidemia_model,
    fit_epidemia,
    prepare_panel,
)
from epidemia.priors import normal, shifted_gamma

# %% [markdown]
# ## Data
#
# The same 11 countries and five interventions both models use.

# %%
ec = epidemia.europe_covid2()
df = ec.data.copy()

# Flaxman's design has SIX columns, not five. The fourth --
#   firstIntervention = 1*((school + selfIsolation + publicEvents
#                           + lockdown + socialDistancing) >= 1)
# (utils/process-covariates.r) -- is an indicator that ANY measure is in force.
# It is the decisive term: it absorbs the common "something changed" drop, so
# the five individual coefficients are left to explain only what distinguishes
# them from one another. Omit it and the collinear five redistribute the whole
# effect among themselves, which is exactly what the epidemia vignette shows.
BASE = list(epidemia.EUROPE_COVID_NPIS)
df["any_intervention"] = (df[BASE].sum(axis=1) >= 1).astype(float)
NPIS = BASE + ["any_intervention"]

panel, series = prepare_panel(
    df, npis=NPIS, responses=["deaths"], pop="pop", fit_until="2020-05-05",
)
obs = [ObsModel("deaths", series["deaths"]["y"], series["deaths"]["mask"],
                i2o=ec.inf2death, family="neg_binom", link_K=0.02)]
print(f"{len(panel.regions)} countries, {panel.X.shape[2]} NPIs, "
      f"{int(panel.lengths.sum())} modelled days")
print("NPI order:", NPIS)

# %% [markdown]
# ## Model A — Flaxman
#
# `link="log"` gives $R_t = \exp(\eta)$, so with $\eta = b_0^{(m)} + \sum_k
# \beta_k X_k + b^{(m)}_{\text{lockdown}} X_{\text{lockdown}}$ we get
# $R_t = \mu_m \exp(\sum_k \beta_k X_k + \ldots)$ with $\mu_m = e^{b_0^{(m)}}$ —
# Flaxman's multiplicative form, with $\beta_k = -\alpha_k$.
#
# The country deviation is restricted to lockdown with
# `sd_slope_fixed=[0, 0, 0, 0, nan]`: zero pins a slope's between-country SD at
# zero (no deviation at all), and `nan` means "estimate this one".

# %%
LOCKDOWN = NPIS.index("lockdown")
slope_sd = [0.0] * len(NPIS)
slope_sd[LOCKDOWN] = np.nan          # the only NPI that varies by country

flaxman_cfg = EpiModelConfig(
    gen=ec.si,
    link="log",                       # mu[m] * exp(-X . alpha)
    intercept=False,
    region_effects=True,
    correlated=False,
    sd_slope_fixed=slope_sd,
    prior_covariates=shifted_gamma(shape=1 / 6, scale=1.0,
                                   shift=np.log(1.05) / 6),
    seed_days=6,
    prior_seeds=normal(0.0, 1.0),
)
flaxman_model = build_epidemia_model(panel, obs, flaxman_cfg)
print(f"{len(flaxman_model.free_RVs)} free parameters")

# %% [markdown]
# ## Model B — the epidemia vignette
#
# Same data, same priors on the coefficients. Two changes: a scaled-logit link
# capping $R_t$ at 6.5, and a country deviation on **every** NPI.

# %%
# The vignette model uses only the five NPIs, so it needs its own panel.
panel5, series5 = prepare_panel(
    df, npis=BASE, responses=["deaths"], pop="pop", fit_until="2020-05-05")
obs5 = [ObsModel("deaths", series5["deaths"]["y"], series5["deaths"]["mask"],
                 i2o=ec.inf2death, family="neg_binom", link_K=0.02)]

vignette_cfg = EpiModelConfig(
    gen=ec.si,
    link="scaled_logit",
    R_link_K=6.5,
    intercept=False,
    region_effects=True,
    correlated=False,
    prior_covariates=shifted_gamma(shape=1 / 6, scale=1.0,
                                   shift=np.log(1.05) / 6),
    seed_days=6,
    prior_seeds=normal(0.0, 1.0),
)

# %% [markdown]
# ## Fitting both

# %%
idata_flaxman = fit_epidemia(panel, obs, flaxman_cfg, draws=1000, tune=1000,
                             chains=4, seed=12345, target_accept=0.95,
                             progress_bar=False)
print(epidemia.sampler_diagnostics(idata_flaxman))

# %%
idata_vignette = fit_epidemia(panel5, obs5, vignette_cfg, draws=1000, tune=1000,
                              chains=4, seed=12345, target_accept=0.95,
                              progress_bar=False)
print(epidemia.sampler_diagnostics(idata_vignette))

# %% [markdown]
# ## The effect sizes
#
# This is the comparison the notebook exists for. Flaxman's constraint should
# push the four non-lockdown effects toward zero.

# %%
labels = ["Schools", "Isolating", "Events", "Distancing", "Lockdown",
          "Any intervention"]
epidemia.plot_effects(idata_flaxman, labels=labels, save=False,
                      title="Flaxman specification: global NPI effects").show()

# %%
epidemia.plot_effects(idata_vignette, labels=labels[:5], save=False,
                      title="epidemia vignette: global NPI effects").show()

# %% [markdown]
# Flaxman's paper reports effects as a **relative reduction in $R_t$**, which for
# the log link is $1 - e^{\beta_k}$. That is the scale of the Nature figure.

# %%
def reduction_table(idata, labels):
    beta = np.asarray(idata.posterior["beta"].stack(s=("chain", "draw")))  # (K, S)
    red = 100.0 * (1.0 - np.exp(beta))
    return pd.DataFrame({
        "intervention": labels,
        "median %": np.percentile(red, 50, axis=1).round(1),
        "5%": np.percentile(red, 5, axis=1).round(1),
        "95%": np.percentile(red, 95, axis=1).round(1),
    })


print("Flaxman specification — relative reduction in R_t (%)")
print(reduction_table(idata_flaxman, labels).to_string(index=False))

# %% [markdown]
# ## Which model does the data prefer?
#
# For nested-ish Bayesian models fitted to the same observations the right tool
# is the **expected log pointwise predictive density**, estimated by PSIS-LOO.
# AIC and BIC are not appropriate here: both assume a single maximum-likelihood
# fit and a countable parameter dimension, and a partially pooled model has
# neither — its effective number of parameters is itself estimated.
#
# `az.compare` ranks by ELPD and reports the standard error of the *difference*,
# which is what decides whether a gap is real.

# %%
for name, model, idata in [("flaxman", flaxman_model, idata_flaxman),
                           ("vignette", build_epidemia_model(panel5, obs5, vignette_cfg),
                            idata_vignette)]:
    with model:
        pm.compute_log_likelihood(idata, progressbar=False)

cmp = az.compare({"Flaxman": idata_flaxman, "epidemia vignette": idata_vignette},
                 ic="loo")
print(cmp.to_string())

# %% [markdown]
# ## And how well does each predict?
#
# ELPD is an in-sample (leave-one-out) criterion. CRPS scores the *predictive
# distribution* against the observations directly, and is the metric the
# forecasting literature uses. Lower is better.

# %%
from epidemia.predict import posterior_predict
from epidemia.scoring import crps

y = np.asarray(obs[0].y)
mask = np.asarray(obs[0].mask)

rows = []
for name, idata in [("Flaxman", idata_flaxman),
                    ("epidemia vignette", idata_vignette)]:
    E = np.asarray(idata.posterior["E_deaths"]).reshape(-1, *y.shape)
    aux = np.asarray(idata.posterior["deaths|aux"]).reshape(-1, 1, 1)
    draws = posterior_predict(E, "neg_binom", aux=aux,
                              rng=np.random.default_rng(0))
    rows.append({
        "model": name,
        "mean CRPS": float(np.mean([
            crps(y[m, mask[m]], draws[:, m, mask[m]].T).mean()
            for m in range(y.shape[0])
        ])),
    })
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## Reading the comparison honestly
#
# Two cautions before drawing a conclusion from the numbers above.
#
# **A better ELPD does not make a model's coefficients more trustworthy.** The
# five interventions were enacted within days of each other, so the individual
# effects are only weakly identified in *either* specification. A model can
# predict deaths well while attributing the reduction to the wrong measure.
# Flaxman's constraint is a modelling *choice* about which attribution to
# prefer, not something the data settle.
#
# **The two are not nested.** They differ in the link as well as in the pooling,
# so an ELPD gap cannot be read as "the extra country deviations were or were
# not worth it".
#
# ## What does *not* map
#
# Two pieces of Flaxman's model have no epidemia expression, and both are
# omitted here rather than approximated:
#
# * `mu[m] ~ normal(3.28, kappa)` puts the hierarchical prior on $R_0$ itself.
#   epidemia's `(1 | country)` puts it on $\log R_0$, so the induced prior on
#   $\mu_m$ is log-normal rather than normal.
# * `ifr_noise[m] ~ normal(1, 0.1)` scales each country's infection fatality
#   ratio. epidemia has no per-country IFR multiplier; writing
#   `deaths ~ 0 + country` would give per-country ascertainment but as a
#   *fixed* effect with a different prior.

# %% [markdown]
# ## Where this does not reproduce (yet)
#
# Measured against Nature Figure 2, this specification gets two of five right:
#
# | intervention | Nature Fig. 2 | here |
# |---|---|---|
# | Schools | ~0% | -0.8% |
# | Isolating | ~0% | 1.6% |
# | Events | ~0% | **54.7%** |
# | Distancing | ~0% | **52.5%** |
# | Lockdown | ~**80%** | **30.3%** |
#
# And the fit is not trustworthy: 146 divergent transitions (3.6%) with an R-hat
# of 1.35, so the point estimates above are not a fair reading of the
# specification either.
#
# The most likely remaining cause is the prior on $R_0$. Flaxman writes
#
# ```stan
# mu[m] ~ normal(3.28, kappa);   kappa ~ normal(0, 0.5);
# ```
#
# an informative prior on $R_0$ **on its natural scale**, tightly centred at
# 3.28. This notebook uses `(1 | country)`, which puts a Gamma-scaled deviation
# on $\log R_0$ — a different distributional shape and far more diffuse. That
# lets the model explain the data with a lower $R_0$ and a correspondingly
# weaker lockdown effect, which is exactly the pattern in the table. Pinning
# $R_0$ near 3.28 is what forces the interventions to account for the whole
# fall in transmission.
#
# Two other pieces still have no epidemia expression: `ifr_noise[m] ~ normal(1,
# 0.1)`, a per-country IFR multiplier; and the exact hierarchical scale
# `lockdown[m] ~ normal(0, gamma)`, which is approximated here by decov.
