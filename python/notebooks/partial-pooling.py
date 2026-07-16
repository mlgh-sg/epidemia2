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
# # Partial Pooling
#
# Python counterpart of the R vignette
# [`partial-pooling`](https://mlgh-sg.com/epidemia2/articles/partial-pooling.html).
# This is a **jupytext** notebook stored as a plain `.py` file (percent format).
#
# In the R package, parameters underlying the reproduction numbers are partially
# pooled through a formula operator, `(expr | factor)` (or `(expr || factor)` for
# independent effects). There is no formula mini-language in the Python port —
# instead the pooling is expressed **directly as hierarchical priors** in the
# PyMC model (`epidemia.multilevel`). This notebook explains the mapping and then
# **demonstrates** the three regimes — no pooling, partial pooling, full pooling
# — fitting each with **MCMC** (NUTS via nutpie), *not* Variational Bayes.

# %%
import numpy as np
import pandas as pd
import arviz as az
from plotnine import aes, geom_hline, geom_pointrange, ggplot, labs, coord_flip, position_dodge

import epidemia as epi
from epidemia.plots import theme_epidemia

# %% [markdown]
# ## From R formulas to hierarchical priors
#
# A term `(expr | factor)` says: the columns of the model matrix parsed from
# `expr` have **separate effects per level of `factor`**, drawn from a **common
# prior** whose parameters are themselves estimated — this is what shares
# information across levels. The table below maps the R formula idioms
# (`R(region, date) ~ ...`) to what the Python model does.
#
# | R formula R.H.S.        | Pooling         | In `epidemia.multilevel` |
# |-------------------------|-----------------|--------------------------|
# | `1 + npi`               | **Full** pooling | one global `beta_npi`, shared by all regions |
# | `1 + npi:region`        | **No** pooling   | an independent effect per region, flat/independent priors |
# | `1 + (0 + npi \| region)` | **Partial** pooling | `beta_npi + b[region]`, with `b[region] ~ N(0, sigma_npi)` and `sigma_npi` estimated |
# | `(npi \| region)`       | Partial pooling, correlated intercept+slope | multivariate-normal region effects |
# | `(npi \|\| region)`      | Partial pooling, **independent** intercept+slope | `b[region,k] ~ N(0, sigma_k)`, one `sigma_k` per column |
#
# The key statistical difference:
#
# * **No pooling** gives each region its own free parameter — noisy where data
#   are scarce (early epidemic, small regions).
# * **Full pooling** forces one shared value — hides genuine between-region
#   variation.
# * **Partial pooling** estimates a common distribution $N(0, \sigma_k)$ and lets
#   each region's effect deviate from the global mean by an amount **shrunk**
#   toward zero; $\sigma_k$ (estimated) controls how much. It interpolates
#   between the two extremes and is what the Europe/COVID example uses.
#
# The `epidemia.multilevel` model implements the last two rows: fixed global
# effects `beta_k` **plus** partially-pooled region deviations `b[region, k]`
# with `b[region, k] ~ N(0, sigma_k)` (the `||`, independent-effects, case).

# %% [markdown]
# ## A worked demonstration
#
# We use a small slice of the Europe/COVID data — three countries and a single
# intervention (`lockdown`) — so the three regimes fit quickly and the shrinkage
# is easy to see. (See the [europe-covid](europe-covid.py) notebook for the full
# 11-country, 5-NPI analysis.)

# %%
ec = epi.europe_covid2()
subset = ec.data[ec.data["country"].isin(["Italy", "Sweden", "Norway"])].copy()
npis = ["lockdown"]
fit = epi.prepare_panel(subset, npis, seed_offset=30, death_threshold=10,
                        fit_until="2020-05-05")
print("regions:", fit.regions, "| days:", dict(zip(fit.regions, fit.lengths.tolist())))

# %% [markdown]
# ### Partial pooling
#
# The default `MultilevelConfig` partially pools the region effects. We estimate
# the country-specific lockdown effect $\beta + b^{(m)}$ for each country and the
# between-country SD $\sigma_\text{lockdown}$.

# %%
config = epi.MultilevelConfig(gen=ec.si, i2o=ec.inf2death, seed_days=6)
idata_pp = epi.fit_multilevel(fit, config, draws=500, tune=1000, chains=4,
                              seed=1, progress_bar=True)
print("divergences:", int(idata_pp.sample_stats["diverging"].sum()))
az.summary(idata_pp, var_names=["beta", "sd"])


# %% [markdown]
# ### No pooling (independent per-country priors)
#
# To contrast, we fit each country **separately** — no shared prior — by running
# the single-country model on each country in turn and reading off its lockdown
# effect. Independent fits are exactly the "no pooling" regime.

# %%
def per_country_effect(idata):
    """Country-specific lockdown effect beta + b[region] from a pooled fit."""
    post = idata.posterior
    beta = np.asarray(post["beta"].sel(npi="lockdown").stack(s=("chain", "draw")))
    out = {}
    for r in post.coords["region"].values:
        b = np.asarray(post["b"].sel(region=r, npi="lockdown").stack(s=("chain", "draw")))
        out[str(r)] = beta + b
    return out


pp_eff = per_country_effect(idata_pp)


# For "no pooling" we fit each country on its own with a wide, independent prior
# by turning off shrinkage (a large fixed between-country SD makes b[region]
# effectively free, i.e. unpooled).
config_nopool = epi.MultilevelConfig(
    gen=ec.si, i2o=ec.inf2death, seed_days=6,
    sd_slope_shape=1e6, sd_scale=1.0,   # huge SD prior => effectively no pooling
)
idata_np = epi.fit_multilevel(fit, config_nopool, draws=500, tune=1000, chains=4,
                              seed=1, progress_bar=True)
np_eff = per_country_effect(idata_np)

# %% [markdown]
# ### Comparison
#
# Plotting the country-specific lockdown effect under partial vs. no pooling
# shows **shrinkage**: partially-pooled estimates are pulled toward the shared
# global mean (and have tighter intervals), most visibly for the country with the
# least information.

# %%
rows = []
for regime, eff in [("partial pooling", pp_eff), ("no pooling", np_eff)]:
    for country, draws in eff.items():
        rows.append({
            "country": country, "regime": regime,
            "median": np.median(draws),
            "lo": np.percentile(draws, 5), "hi": np.percentile(draws, 95),
        })
comp = pd.DataFrame(rows)
(
    ggplot(comp, aes("country", "median", color="regime"))
    + geom_hline(yintercept=0.0, linetype="dotted", color="#555555")
    + geom_pointrange(aes(ymin="lo", ymax="hi"), position=position_dodge(width=0.4))
    + coord_flip()
    + labs(x="", y="Lockdown effect on logit $R_t$  ($\\beta + b^{(m)}$)",
           title="Partial vs. no pooling: country-specific lockdown effect")
    + theme_epidemia()
)

# %% [markdown]
# The partially-pooled intervals are narrower and drawn toward one another,
# because information is shared across countries through the common prior
# $N(0, \sigma_\text{lockdown})$ — exactly the behaviour the `(lockdown ||
# country)` term encodes in R. Full pooling would collapse all three countries to
# the single global $\beta$ (set the between-country SD to ~0 to see this).
#
# ### References
#
# * Bates, D. et al. (2015). *Fitting Linear Mixed-Effects Models Using lme4.*
# * Flaxman, S. et al. (2020). *Estimating the effects of non-pharmaceutical
#   interventions on COVID-19 in Europe.*
