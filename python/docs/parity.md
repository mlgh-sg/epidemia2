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
| Non-centred parameterisation | ✅ | ✅ |
| Deterministic renewal process | ✅ | ✅ |
| Hierarchical seeding | ✅ | ✅ |
| Covariates on `R_t` | ✅ formula | ⚠️ design matrix |
| Counterfactuals | ✅ | ⚠️ `R_t` only |
| Posterior predictive | ✅ | ⚠️ mean only |
| Correlated random effects | ✅ | ❌ |
| Random walk on `R_t` in a multi-region model | ✅ | ❌ |
| Multiple observation series | ✅ | ❌ |
| Time-varying ascertainment | ✅ | ❌ |
| Stochastic latent infections | ✅ | ❌ |
| Population / susceptibility adjustment | ✅ | ❌ |
| Forecasting from new data | ✅ | ❌ |
| Forecast scoring (CRPS, coverage) | ✅ | ❌ |
| Variational inference | ✅ | ❌ |
| Swappable prior families | ✅ | ❌ |
| nutpie sampler, JAX/GPU option | ❌ | ✅ |
| `plotnine` plots | ❌ | ✅ |

## The three that will bite you first

**A time-varying `R_t` across regions.** Almost every R tutorial combines a random
walk with grouped data — `R(region, date) ~ 1 + rw(time = dt)`. Here the two
features live in different models: `build_model` has the walk but is
single-population, and `build_multilevel_model` is multi-region but its `R_t` is a
deterministic step function of the covariates. There is no way to write the
combination.

**Several observation series at once.** Two R vignettes fit deaths and cases
jointly, each with its own delay distribution, family, link and priors. Both
Python builders declare exactly one observed variable, so `EuropeCovid2`'s `cases`
column is never used.

**Population adjustment.** Neither recursion carries a susceptible pool, so
long-horizon infections grow without bound. Fine over the eight-week windows the
notebooks use; wrong for anything longer.

## The partial ones, precisely

**Covariates** are supplied as a numeric `(regions, time, covariates)` array from
`prepare_panel`, not as a formula. You get covariates; you write the design matrix
yourself.

**Posterior predictive** — `plots.plot_obs` bands the posterior of the expected
count `E_deaths`. It does not draw from the negative binomial on top, so the
ribbons are narrower than R's `plot_obs`, which does. They answer different
questions: "where is the mean" versus "where would an observation fall".

**Counterfactuals** are available for effect sizes through `effect_table`, which
re-evaluates `R_t` per draw under a modified design. Re-simulating *observations*
under that design is notebook code, not a library function.

## Choosing a side

Use **R** for anything you intend to publish from: several data streams,
forecasting and scoring, susceptibility, the full prior system.

Use **Python** when the model is a partially pooled multi-region renewal model
with fixed covariates and you want it in a PyMC/ArviZ workflow. See
[Performance](performance.md) for how the two backends actually compare on that
model — the answer is less one-sided than "Rust sampler beats Stan" suggests.
