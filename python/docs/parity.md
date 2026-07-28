# What this port does and does not do

`epidemia` exists twice in this repository: the **R package** at the repository
root is the reference implementation, and this **Python package** is a focused
port of its core. The two share a model, not a feature set.

This page is the honest inventory. It is checked against the code rather than
against intentions, because the most expensive way to learn that a feature is
missing is halfway through an analysis.

## At a glance

| Capability | R | Python |
|---|:--:|:--:|
| Multiple regions / groups | ✅ | ✅ |
| Multilevel / partial pooling | ✅ | ✅ |
| Correlated random effects | ✅ | ✅ `correlated=True` |
| Random walk on `R_t`, incl. per region | ✅ | ✅ `RandomWalk(by_region=)` |
| Multiple observation series | ✅ | ✅ `[ObsModel, ObsModel]` |
| Time-varying ascertainment | ✅ | ✅ `ObsModel(X=)` |
| Population / susceptibility adjustment | ✅ | ✅ `pop_adjust=True` |
| Observation families | ✅ 5 | ✅ 5 |
| Forecast scoring (CRPS, coverage) | ✅ | ✅ `epidemia.scoring` |
| Posterior predictive sampling | ✅ | ✅ `epidemia.predict` |
| Plots band the posterior **predictive** | ✅ | ✅ `plot_obs(predictive=True)`, default |
| Spaghetti plots | ✅ | ✅ `spaghetti_rt` etc. |
| Variational inference | ✅ | ✅ `fit_variational` |
| Swappable prior families | ✅ | ✅ `epidemia.priors` |
| Hierarchical seeding | ✅ | ✅ |
| Non-centred parameterisation | ✅ | ✅ |
| Covariates on `R_t` | ✅ formula | ✅ `parse_formula` / `build_from_formula` |
| Forecasting from new data | ✅ one call | ✅ `forecast(...)` |
| Random walk keeps walking over the horizon | ✅ | ✅ `forecast(rw_forecast="draw")`, default |
| Stochastic latent infections | ✅ | ✅ `latent=True` |
| Autoscaled priors, `hs`/`lasso` families | ✅ | ✅ applied by the builder; `EpiModelConfig(autoscale=)` |
| Counterfactuals | ✅ | ⚠️ `R_t` via `effect_table`; observations via `forecast` |
| Observation-level random walk | ✅ | ✅ `ObsModel(rw=)` |
| Vaccination removal (`epiinf(rm=)`) | ✅ | ✅ `EpiModelConfig(rm=, prior_rm_noise=)` |
| Prior predictive (`prior_PD`) | ✅ | ✅ `EpiModelConfig(prior_PD=True)` |
| `group_subset` | ✅ | ✅ `prepare_panel(group_subset=)` |
| `posterior_linpred` | ✅ | ✅ `epidemia.postprocess` |
| `posterior_infectious` (incl. `/ max(gen)`) | ✅ | ✅ `epidemia.postprocess` |
| `prior_summary` | ✅ | ✅ `epidemia.postprocess` |
| `as.matrix` / `get_samps` | ✅ | ✅ `extract_samples` |
| Link functions | ✅ | ✅ 3 on `R_t`, 6 on observations |
| `center=` | ✅ | ✅ |
| Fully pooled (no region effects) | ✅ | ✅ `region_effects=False` |
| `plot_coverage` / `plot_metrics` | ✅ | ✅ `epidemia.plots` |
| `plot_infectious` / `plot_linpred` | ✅ | ✅ `epidemia.plots` |
| Forecasting a latent (`latent=True`) fit | ✅ | ✅ drawn, not frozen |
| ADVI stops on ELBO convergence | ✅ `tol_rel_obj` | ✅ `fit_variational(early_stop=)` |
| Default prior scale when unspecified | 0.25 | 0.25 |
| `lkj()` as `prior_covariance` | ❌ rejected (no Stan support) | ✅ proper LKJ covariance |
| Exclude days from a random walk | ✅ `NA` in `rw(time=)` | ✅ `index = -1` |
| Sampler diagnostics retained on the fit | ✅ `sampler_diagnostics()` | ✅ `sampler_diagnostics()` |
| nutpie sampler, JAX/GPU option | ❌ | ✅ |
| `plotnine` plots | ❌ | ✅ |

Everything in the top block used to be missing. The model that expresses it is
:func:`epidemia.build_epidemia_model`, which supersedes the two earlier builders:
`build_model` had a random walk but one population, `build_multilevel_model` had
regions but a deterministic `R_t` and a single series, and neither could write
what the R tutorials write routinely.

```python
from epidemia.core import (EpiModelConfig, ObsModel, PanelData,
                           RandomWalk, build_epidemia_model, prepare_panel)

panel, series = prepare_panel(
    df, npis=["lockdown"], responses=["deaths", "cases"],
    pop="pop", rw_by="week", fit_until="2020-05-05")

model = build_epidemia_model(
    panel,
    [ObsModel("deaths", **series["deaths"], i2o=inf2death,
              family="neg_binom", link_K=0.02),
     ObsModel("cases", **series["cases"], i2o=inf2case,
              family="quasi_poisson", link_K=0.4)],
    EpiModelConfig(gen=si, correlated=True, pop_adjust=True,
                   rw=RandomWalk(index=panel.rw_index, by_region=True)),
)
```

That is R's `R(country, date) ~ (1 + lockdown | country) + rw(time = week, gr = country)`
with `obs = list(deaths, cases)` and `epiinf(pop_adjust = TRUE)`.

## Writing a model as a formula

```python
from epidemia.formula import build_from_formula
from epidemia.core import EpiModelConfig, ObsModel, fit_epidemia

panel, series, cfg = build_from_formula(
    df,
    "R(country, date) ~ 1 + rw(time = week, gr = country) + lockdown",
    responses=["deaths", "cases"], pop="pop", fit_until="2020-05-05")

idata = fit_epidemia(panel, obs_models, EpiModelConfig(gen=si, **cfg))
```

`||` gives independent region effects and `|` correlated ones; `0 +` drops the
intercept; `rw(gr = x)` gives one walk per level of `x` and `rw()` one shared
walk. Non-numeric covariates are dummy-coded with R's treatment contrasts.

## Forecasting

```python
from epidemia.forecast import forecast

fc = forecast(idata, panel, obs_models, config, newdata=longer_df)
fc.to_frame()          # tidy long frame: region, date, variable, quantiles
```

Nothing is re-fitted: conditional on a posterior draw the latent series are
deterministic, so this is forward simulation over draws. Covariates past the fit
window carry forward, and the random walk is **held at its final fitted step** —
the forecast says "`R_t` stays where it ended" rather than inventing increments.
`fc.predicted` holds draws from the observation family, not just the mean.

## The caveats that remain

**The formula's fixed/random split is advisory.** `build_epidemia_model` shares
one design matrix between the population-level `beta` and the region-level `b`,
so a covariate appearing only inside `(... | group)` still gets a population
coefficient, and vice versa. The *pooling* structure — `|` versus `||`, whether
there is an intercept — is exact; the column split is not. Nesting
(`(1 | county/district)`) and interactions (`a:b`) are rejected rather than
silently mis-parsed, and only one `rw()` term is allowed.

**Forecasting does not support `latent=True`.** With latent infections the
post-seeding infections are free parameters, so there is no forward-simulable
rule past the fitted window. Use the deterministic recursion to forecast.

**Counterfactuals on observations** are now possible by editing the covariates in
`newdata` and calling `forecast`, but there is no dedicated helper as in R.

## Choosing a side

The two now express the same models. Use **R** when you want the formula
interface, one-call forecasting from a `newdata` frame, or stochastic latent
infections. Use **Python** when you want a PyMC/ArviZ workflow, nutpie's
sampler, or to build the design matrices yourself. See
[Performance](performance.md) for how the two backends actually compare on that
model — the answer is less one-sided than "Rust sampler beats Stan" suggests.
