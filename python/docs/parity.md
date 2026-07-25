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
| Spaghetti plots | ✅ | ✅ `spaghetti_rt` etc. |
| Variational inference | ✅ | ✅ `fit_variational` |
| Swappable prior families | ✅ | ✅ `epidemia.priors` |
| Hierarchical seeding | ✅ | ✅ |
| Non-centred parameterisation | ✅ | ✅ |
| Covariates on `R_t` | ✅ formula | ⚠️ design matrix |
| Forecasting from new data | ✅ one call | ⚠️ primitives, no wrapper |
| Counterfactuals | ✅ | ⚠️ `R_t` only |
| Stochastic latent infections | ✅ | ❌ |
| Autoscaled priors, `hs`/`lasso` families | ✅ | ❌ |
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

## What is still not here

**A formula interface.** Covariates come in as a numeric
`(regions, time, covariates)` array from `prepare_panel`; there is no
`R(country, date) ~ ...` parser. You get the models, you write the design matrix.

**Forecasting is primitives, not one call.** `epidemia.predict` gives you
`simulate` (forward renewal over posterior draws, with the susceptibility
adjustment), `expected_observations` and `posterior_predict` (which draws from
the observation family rather than banding the mean). What is missing is R's
convenience of handing a whole `newdata` frame to one function.

**Stochastic latent infections.** R's `epiinf(latent = TRUE)` treats infections
as parameters with noise around the renewal equation. Here the recursion stays
deterministic.

**Prior coverage is close but not complete.** `epidemia.priors` mirrors R's
constructors — `normal`, `student_t`, `cauchy`, `exponential`, `laplace`,
`shifted_gamma`, `hexp`, `decov`, `lkj` — with R's argument names and
validation. Not ported: `autoscale`, and the `hs`/`hs_plus`/`lasso`/
`product_normal` shrinkage families.

**Counterfactuals** cover effect sizes through `effect_table`, which re-evaluates
`R_t` per draw under a modified design. Re-simulating observations under that
design is now possible with `epidemia.predict`, but it is your code, not a
one-liner.

## Choosing a side

The two now express the same models. Use **R** when you want the formula
interface, one-call forecasting from a `newdata` frame, or stochastic latent
infections. Use **Python** when you want a PyMC/ArviZ workflow, nutpie's
sampler, or to build the design matrices yourself. See
[Performance](performance.md) for how the two backends actually compare on that
model — the answer is less one-sided than "Rust sampler beats Stan" suggests.
