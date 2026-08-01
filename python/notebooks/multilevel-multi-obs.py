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
# # Multilevel models with several observation series
#
# The Python counterpart of the R vignette
# [`multilevel-multi-obs`](https://mlgh-sg.com/epidemia2/articles/multilevel-multi-obs.html).
#
# This is the model the earlier Python builders could not express. It puts four
# things together at once:
#
# * **partial pooling** of reproduction numbers across regions, with the region
#   intercept and slope *correlated* (R's `(1 + x | region)` with `decov`);
# * a **random walk** on $R_t$, one per region (R's `rw(time = week, gr = region)`);
# * **two observation series** fitted jointly, deaths and cases, each with its own
#   delay distribution, family and ascertainment rate;
# * a **susceptibility adjustment**, so infections saturate instead of growing
#   without bound.
#
# All of it comes from `epidemia.core`.

# %%
import numpy as np
import pandas as pd
import arviz as az

import epidemia
from epidemia.core import (
    EpiModelConfig,
    ObsModel,
    RandomWalk,
    build_epidemia_model,
    fit_epidemia,
    prepare_panel,
)

# %% [markdown]
# ## Data
#
# `EuropeCovid2` carries daily deaths *and* daily cases for eleven countries,
# alongside the intervention indicators. We use all eleven countries, and add a
# month column to index the random walk.

# %%
ec = epidemia.europe_covid2()
df = ec.data.copy()
# A monthly walk index, not a weekly one. See "Read the diagnostics" below:
# a walk free to move every week can absorb a one-off step covariate into its
# own increments, leaving beta[lockdown] only weakly identified.
df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")

panel, series = prepare_panel(
    df,
    npis=["lockdown"],
    responses=["deaths", "cases"],
    pop="pop",
    rw_by="month",
    fit_until="2020-05-05",
)
print(panel.regions, panel.X.shape, panel.pops.astype(int))

# %% [markdown]
# `prepare_panel` windows every response onto one shared time axis: each region
# starts 30 days before its tenth cumulative death, as in the R vignettes. Series
# observed on different days are handled by their own masks, so a weekly survey
# and a daily count can sit in the same model.
#
# ## The two observation series
#
# Neither series *is* the infections. Each is infections convolved with a delay
# distribution and scaled by an ascertainment rate that is itself estimated:
#
# $$\mathbb{E}\left[Y^{(s)}_{t,m}\right]
#    = \alpha^{(s)}_{t,m} \sum_{k \ge 1} \pi^{(s)}_k\, i_{t-k,m}$$
#
# For deaths, $\alpha$ is the infection fatality ratio, capped at 2% by
# `link_K`. For cases it is the ascertainment ratio, capped at 40%. The
# infection-to-case delay here is deliberately crude — undetected for four days,
# then uniform over a week.

# %%
i2o_cases = np.concatenate([np.zeros(4), np.full(7, 1 / 7)])

obs = [
    ObsModel(
        "deaths", series["deaths"]["y"], series["deaths"]["mask"],
        i2o=ec.inf2death, family="neg_binom", link_K=0.02,
    ),
    ObsModel(
        "cases", series["cases"]["y"], series["cases"]["mask"],
        i2o=i2o_cases, family="quasi_poisson", link_K=0.4,
    ),
]

# %% [markdown]
# ## The transmission model
#
# `correlated=True` estimates the full covariance of the region intercept and the
# lockdown slope through an LKJ-Cholesky prior — R's single-bar `(1 + lockdown |
# country)`. Setting it `False` gives the independent double-bar form.
#
# `RandomWalk(by_region=True)` gives each country its own monthly walk with its
# own scale; `by_region=False` would put them all on one shared walk.
#
# `pop_adjust=True` tracks the susceptible pool.

# %%
config = EpiModelConfig(
    gen=ec.si,
    correlated=True,
    pop_adjust=True,
    prior_susc_mean=0.95,
    prior_susc_sd=0.05,
    rw=RandomWalk(index=panel.rw_index, by_region=True, prior_scale=0.1),
)

model = build_epidemia_model(panel, obs, config)
print(f"{len(model.free_RVs)} free parameters")

# %% [markdown]
# ## Fitting
#
# `fit_epidemia` builds and samples in one call. Two of its defaults differ from
# nutpie's, both because of this model class's geometry: `target_accept=0.95` for
# the funnel between the between-region SDs and the non-centred effects, and
# `adaptation="low_rank"` for the correlated ridge that the covariates and the
# random walk create together. A diagonal mass matrix cannot follow that ridge,
# and it fails *silently* — bad mixing with no divergences to warn you. See the
# [performance](../performance.md) page for the measurements.

# %%
idata = fit_epidemia(panel, obs, config, draws=1000, tune=1000, chains=4,
                     seed=12345, target_accept=0.99, progress_bar=False)
summ = az.summary(idata, var_names=["beta", "seed", "rw_scale",
                                   "Sigma_chol_stds"])
print(summ[["mean", "sd", "r_hat", "ess_bulk"]].to_string())

# %% [markdown]
# ### Read the diagnostics before the estimates
#
# Check the sampler before reading a single estimate. `sampler_diagnostics()`
# is the mirror of R's function of the same name and reports the same
# quantities: divergent transitions, iterations that saturated the maximum tree
# depth, E-BFMI per chain, and the worst R-hat and effective sample size across
# all parameters.

# %%
print(epidemia.sampler_diagnostics(idata))

# %% [markdown]
# The three sampler quantities answer different questions. **Divergent
# transitions** mean the sampler could not follow the posterior's curvature;
# they bias the result and drawing more samples does not help, so they have to
# be fixed by raising `target_accept` or reparameterising. **Max treedepth** is
# an efficiency problem rather than a correctness one. **E-BFMI** below about
# 0.2 suggests the momentum resampling is not exploring the energy distribution.
#
# Two choices earlier in this notebook were made to keep these numbers clean,
# and both are worth understanding because they are easy to get wrong.
#
# **The walk is monthly, not weekly.** A random walk and a one-off step
# covariate compete for the same signal: lockdown is a single permanent change
# in one direction, and a walk free to move every week can absorb exactly that
# change into its own increments. The two are then only weakly separable, the
# posterior has a ridge along which `beta[lockdown]` trades off against the
# walk, and the chains explore it slowly — a low `ess_bulk` and an `r_hat` above
# 1.01 for that one coefficient. A monthly index cannot track a weekly step, so
# the competition goes away. This is not a Python quirk; the same formula in R
# (`~ lockdown + rw(time = week, gr = country)`) has the same problem.
#
# **All eleven countries, not a handful.** A partially pooled slope asks the
# data to estimate a *variance* across groups, and a few groups barely identify
# it. Cutting the panel down to three countries to save runtime reintroduces
# divergences and a tail ESS in the tens.
#
# If you do hit this in your own model, the honest responses are to report the
# parameter as poorly identified and not quote its interval, to drop one of the
# two competing terms, or to make the walk coarser as done here.

# %%
bad = summ[(summ["r_hat"] > 1.01) | (summ["ess_bulk"] < 400)]
if len(bad):
    print("weakly identified or poorly mixed:")
    print(bad[["mean", "ess_bulk", "r_hat"]].to_string())
else:
    print(f"all clear: max r_hat = {summ['r_hat'].max():.3f}")

# %% [markdown]
# ## Both series are explained
#
# A joint model has to fit everything it is conditioned on, so check each series.

# %%
# `plot_obs` bands the posterior **predictive** -- draws pushed through each
# series' observation family -- so the ribbon carries the negative-binomial (and
# quasi-Poisson) noise, not just parameter uncertainty. Banding `E_deaths`
# directly, which an earlier version of this notebook did by hand, gives an
# interval several times too narrow to compare against the counts drawn over it.
#
# Three nested credible bands, as R's `plot_obs` draws by default.

# %%
for name, model in zip(("deaths", "cases"), obs):
    epidemia.plot_obs(
        idata, data=panel, obs_model=model, series=name,
        levels=(30, 60, 90), ylab=name.capitalize(),
        title=f"Posterior predictive: {name}", save=f"multilevel-obs-{name}",
    )

# %% [markdown]
# ## What the susceptibility adjustment does
#
# With `pop_adjust`, the realised $R_t$ is the unadjusted rate scaled by the
# susceptible fraction, so it falls as the epidemic proceeds even when the
# unadjusted rate is flat. Over an eight-week window the effect is small; over a
# long forecast it is what stops infections growing without bound.

# %%
S = idata.posterior["susceptible"].mean(("chain", "draw")).values
Rt = idata.posterior["Rt"].mean(("chain", "draw")).values
Rt_un = idata.posterior["Rt_unadj"].mean(("chain", "draw")).values

for m, region in enumerate(panel.regions):
    n = int(panel.lengths[m])
    print(f"{region:10s} susceptible {S[m, n-1] / panel.pops[m]:.3f} "
          f"at the end;  Rt {Rt[m, n-1]:.2f} vs unadjusted {Rt_un[m, n-1]:.2f}")

# %% [markdown]
# ## Per-region reproduction numbers
#
# Each country has its own weekly walk, so $R_t$ varies smoothly within a country
# rather than stepping only when a policy changes.

# %%
epidemia.plot_rt(
    idata, data=panel, levels=(30, 60, 90),
    title="Reproduction numbers, with a monthly walk per region",
    save="multilevel-rt",
)

# %% [markdown]
# ## Scoring the fit
#
# `epidemia.scoring` mirrors R's `evaluate_forecast`. Feed it observations and
# predictive draws — note the orientation: one **row per observation**, one column
# per draw.
#
# Note also that these are draws from the posterior *predictive*, not bands on the
# expected count: `epidemia.predict.posterior_predict` samples from the negative
# binomial, so the intervals include observation noise. Banding `E_deaths` alone
# would give intervals that are too narrow.

# %%
from epidemia.predict import posterior_predict
from epidemia.scoring import evaluate_forecast

m = 0                                  # score the first region
n = int(panel.lengths[m])
E = idata.posterior["E_deaths"].isel(region=m).values.reshape(-1, panel.X.shape[1])[:, :n]
aux = idata.posterior["deaths|aux"].values.reshape(-1)

pp = posterior_predict(E, family="neg_binom", aux=aux[:, None], rng=np.random.default_rng(0))

mask = series["deaths"]["mask"][m, :n]
y = series["deaths"]["y"][m, :n]
result = evaluate_forecast(
    y[mask], pp[:, mask].T,                     # (observations, draws)
    date=pd.to_datetime(panel.dates[m][:n])[mask],
    levels=(50, 95),
)
print(result.error.head())
print(result.coverage.groupby("level")["in_ci"].mean().round(3))

# %% [markdown]
# ## Where this differs from R
#
# The models are the same; the interfaces are not. R parses
# `R(country, date) ~ (1 + lockdown | country) + rw(time = week, gr = country)`
# from a formula, while here you hand `prepare_panel` the column names and it
# builds the design matrix. Forecasting from a `newdata` frame is one call in R;
# here you use `epidemia.predict.simulate` yourself. See
# [parity](../parity.md) for the full list.
