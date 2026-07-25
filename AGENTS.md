# AGENTS.md — working on the epidemia codebase

Guidance for AI agents (and humans) modifying this package. For *using* the R
package to build epidemic models, see `llms.txt` and the vignettes; for the
Python port, see `python/README.md` and `python/docs/` (`python/docs/llms.txt`).

## Two implementations — keep them in parity

This repository ships **two implementations of the same model**, on one branch:

- **R (repo root)** — the **reference** implementation. Feature-complete: the full
  `epirt`/`epiinf`/`epiobs`/`epim` API, covariate formulas, multilevel/partial
  pooling, multiple joint observation series, latent infections, population/
  susceptibility adjustment, the full prior system, forecasting via `newdata`,
  variational Bayes, and rich post-processing. Backend: **CmdStanR**.
- **Python (`python/`)** — a **focused, fast core** port. Single-population renewal
  model (`EpiConfig` + `build_model`), non-centred RW on `R_t`, log / scaled-logit
  links, Poisson / neg-binom / normal families, `plotnine` plots, a NumPy renewal
  reference. Backend: **PyMC + nutpie** (numba CPU, or JAX for GPU on Linux).

What is **extra in each**. Verified against the code (not aspirational) — if you
change either side, re-check the row.

| Capability | R (root) | Python (`python/`) |
|---|---|---|
| Multiple regions / groups | ✅ | ✅ `prepare_panel` → `build_multilevel_model` |
| Multilevel / partial pooling across groups | ✅ | ✅ `multilevel.py`, non-centred, 3 pooling regimes |
| Covariates for `R_t` | ✅ formula mini-language | ⚠️ numeric `(M,T,K)` design matrix; no formula parser |
| Correlated random effects (`(x \| g)` + `decov`) | ✅ | ❌ independent (`\|\|`) only, no covariance matrix |
| Random walk on `R_t` | ✅ incl. `rw(gr=)` per group | ⚠️ single-population only; disjoint from the multilevel model |
| Multiple observation series (joint) | ✅ up to 10 | ❌ exactly one likelihood |
| Time-varying ascertainment (obs-level RW) | ✅ | ❌ scalar IFR only |
| Latent (stochastic) infections | ✅ `epiinf(latent=TRUE)` | ❌ deterministic renewal only |
| Population / susceptibility adjustment | ✅ | ❌ no susceptible pool |
| Full prior system (swappable families) | ✅ | ❌ families hardcoded; hyperparameters configurable |
| `shifted_gamma`, `hexp` seeding | ✅ | ⚠️ built in to the multilevel model, not reusable priors |
| Forecasting via `newdata` | ✅ | ❌ no `predict`; notebook code only |
| Counterfactuals | ✅ | ⚠️ `effect_table` does them for `R_t` only |
| Forecast scoring (CRPS / coverage) | ✅ | ❌ |
| Variational inference | ✅ | ❌ NUTS only (`adaptation` picks a metric, not an algorithm) |
| Posterior predictive | ✅ sampled | ⚠️ bands `E_obs`; excludes observation noise |
| Spaghetti (per-draw) plots | ✅ | ❌ draws always collapsed to intervals |
| Non-centred RW parameterisation | ✅ | ✅ |
| nutpie sampler + JAX/GPU backend option | ❌ | ✅ |
| Grammar-of-graphics (`plotnine`) plots | ❌ (ggplot2 via R) | ✅ |

The three gaps that most affect someone following the R tutorials, in order:
**a time-varying `R_t` in a multi-region model** (R's tutorials routinely combine
`rw()` with grouped data; Python's walk is single-population and its multilevel
`R_t` is a deterministic covariate step function), **multiple joint observation
series** (two whole vignettes rest on it), and **population adjustment** (without
it long-horizon infections grow unbounded).

For measured performance of the two backends on the same model, see
`benchmarks/` and the "Performance" page of the Python docs.

**The rule: a model change to one language should be ported to the other.** When you
add or change modelling behaviour (a new family, link, prior, the RW
parameterisation, the renewal/observation recursion, seeding, …) in one
implementation, port the equivalent change to the other so they stay in parity —
and if a full port is out of scope for the change, **use agentic coding assistance
(e.g. Claude Code) to do the port and integrate it**, then update the table above.
Keep the two renewal cores mathematically identical: R's `inst/stan/*` recursion and
Python's `python/src/epidemia/model.py` (PyTensor) + `renewal.py` (NumPy reference)
must agree. If you intentionally leave a feature in only one language, say so in the
table rather than letting the two silently drift.

## What this package is

`epidemia` fits Bayesian **semi-mechanistic** models of infectious diseases. Latent
daily infections are propagated by a discrete **renewal process** (a self-exciting
point process); the time-varying reproduction number `R_t` and observation
ascertainment rates are modelled as (possibly multilevel) regressions. Inference is
done in **Stan**. Software paper: Scott et al. (2021), *Epidemia: An R Package for Semi-Mechanistic Bayesian Modelling of Infectious Diseases using Point Processes*, arXiv:2110.12461. Applied companion: Mishra et al. (2022), *A COVID-19 Model for Local Authorities of the United Kingdom*, JRSS-A, doi:10.1111/rssa.12988.

## Toolchain (important)

- Models are fit with **CmdStanR** (`cmdstanr`) + **CmdStan**, NOT rstan. Draws are
  represented with the **posterior** package. `rstan` and `rstanarm` are NOT
  dependencies.
- Stan programs are compiled **on first use and cached** in
  `tools::R_user_dir("epidemia", "cache")`. There is no install-time compilation
  (no `src/`, `configure`, or `LinkingTo`).
- Requires CmdStan ≥ 2.33 (developed against 2.36). Install once with
  `cmdstanr::install_cmdstan()`.

## Environment setup

The R environment is locked with **renv** (`renv.lock`), and CmdStan is pinned
separately at **2.36.0** — renv only manages R packages, and CmdStan is a C++
toolchain installed outside any R library.

```sh
make setup     # renv::restore() + install CmdStan 2.36.0 if absent
```

That is `Rscript tools/setup.R` if you would rather not use make. The project
`.Rprofile` activates renv and puts the Stan r-universe on `getOption("repos")`
so `cmdstanr` resolves.

To rebuild the lockfile from scratch after changing `DESCRIPTION`, run
`Rscript tools/renv-bootstrap.R`. Two things about that script are load-bearing:
the snapshot type is **implicit** (renv scans the code) because an explicit
snapshot would record only `Imports` and miss everything the tests and tutorials
need; and `tools/dependencies.R` lists the handful of packages the scan cannot
see, because they are used only inside `include = FALSE` chunks that
`precompute.R` strips when baking the tutorials.

The dev tooling (`roxygen2`, `devtools`, `pkgdown`, `covr`) is in the lockfile
too, via that same file — it is referenced only from the `Makefile`, which renv
does not scan, so `make document` and `make docs` would otherwise break on a
bare restore.

## Common commands

```sh
make test        # testthat suite, fast tier
make test-slow   # ...including the heavier fitting tests (EPIDEMIA_SLOW_TESTS=true)
make document    # roxygen2::roxygenise() -> NAMESPACE + man/
make compile     # precompile both Stan programs into the user cache
make tutorials   # re-bake the precomputed tutorial vignettes
make check       # R CMD check
```

Tests run against the working tree via `pkgload::load_all(".")`, not an installed
copy. `make tutorials-clean` drops `vignettes/cache/` first — without that, knitr
replays the `cache = TRUE` chunks and the tutorials are not actually refitted.

Do not edit `NAMESPACE` or `man/*.Rd` by hand — they are generated by roxygen2 from
`#'` blocks in `R/`.

## Architecture / where things live

| Area | Files |
|---|---|
| Model specification API | `R/epirt.R`, `R/epiinf.R`, `R/epiobs.R`, `R/epim.R` |
| Prior constructors | `R/priors.R` (vendored from rstanarm), `R/additional_priors.R` (`shifted_gamma`, `hexp`) |
| Stan compilation + caching | `R/stanmodels.R` (`epidemia_stan_model()`, `compile_epidemia()`) |
| **CmdStanR backend** | `R/backend.R` — fitting, arg translation, the `epimodel_draws` wrapper, summaries, sparse-parts |
| Standata construction | `R/stan_data.R`, `R/standata_reg.R`, `R/parse_mm.R` |
| Fitted-object + methods | `R/epimodel.R`, `R/as.matrix.epimodel.R`, `R/summary.R`, `R/print.R`, `R/misc.R` |
| Posterior / prediction | `R/posterior_*.R`, `R/pp_eta.R`, `R/forecasting.R` |
| Plotting | `R/plots.R`, `R/plots_epi.R`, `R/geom_stepribbon.R` |
| Stan programs | `inst/stan/epidemia_base.stan` (fitting), `inst/stan/epidemia_pp_base.stan` (generated quantities), with modular `#include`s under `inst/stan/*/` |

## How a fit flows

`epim()` → build model matrices (`epirt_`/`epiobs_`) → `standata_all()` → `fit_cmdstan()`
(`$sample()`/`$variational()`) → `build_draws()` (posterior draws behind the
`epimodel_draws` wrapper, with `theta_L`→`Sigma` transformed) → `epimodel()` object.
Latent series (`posterior_rt`, `posterior_infections`, `posterior_predict`) are
produced by running `inst/stan/epidemia_pp_base.stan` as CmdStan **generate
quantities** over the posterior draws (`R/posterior_sims.R`).

## Landmines (read before editing these areas)

- **`as.matrix.epimodel_draws` must return a plain base matrix.** Leaving posterior
  S3/S4 classes on it makes `draws_matrix %*% <sparse Z>` (random-effect / random-walk
  linear predictors in `pp_eta.R`) recurse to a C-stack overflow.
- **generate-quantities input is strict.** The fitted-parameter draws passed to
  `epidemia_pp_base` must be *exactly* that model's parameters, in declaration order.
  See the subset+reorder in `R/posterior_sims.R`. `pp_stanmat()` pads each vector to
  the pp model's `+2` dimensions.
- **`clean_standata(sdat, model_data_vars(mod))`** restricts data to the variables
  the Stan program declares — this drops R-only bookkeeping fields (e.g. `groups`,
  `prior_dist_name`) that CmdStanR would otherwise reject.
- **Sparse CSR indices are 0-based** throughout (`standata_reg.R` stores `parts$v - 1L`).
  The pure-Stan `csr_matrix_times_vector2` in `inst/stan/functions/common_functions.stan`
  matches that convention.
- **Stan syntax** must be canonical (CmdStan ≥ 2.33): `array[n] int x`, and
  `foo_cdf(y | ...)` with a vertical bar.
- The set of usable prior *families* is fixed by the Stan model, not R — see
  `ok_dists`/`ok_int_dists`/`ok_aux_dists`/`ok_cov_dists` in `R/utilities.R`. Adding a
  new family means adding an integer code branch in `inst/stan/model/priors_*.stan`
  and the R parser, plus a constructor in `R/priors.R`. **`lkj()` is exported (it
  comes with the vendored rstanarm prior set) but is not usable as
  `prior_covariance`:** the Stan program has only `decov_lp()`. `epirt()` rejects
  it explicitly, because letting it through produced standata with `shape = 0`
  and a fit that died inside CmdStan after compiling.
- **Random effects are R_t-only.** `epirt()` takes `prior_covariance`; `epiobs()`
  has no such argument. The Stan program declares a single `z_b`/`b`/`theta_L`
  set, consumed by `tparameters/make_eta.stan` and never by `make_oeta.stan`. A
  formula like `deaths ~ (1|region)` passed to `epiobs()` therefore will not do
  what it looks like it does.
- **Forecasting cannot introduce a group.** Passing `newdata` containing a group
  the model was not fitted on silently drops that group rather than predicting
  for it. The `_NEW_` level added by `pad_reTrms()` (`R/helpers.R`) exists for
  the internal design-matrix padding, not for out-of-sample groups.
- **Fitted-model fixtures under `tests/data/` are coupled to the draws
  representation** and go stale silently. Regenerate them with
  `Rscript tests/data/make-fixtures.R` after changing `build_draws()` or the
  `epimodel_draws` wrapper.

## Conventions

- Style: match surrounding code (2-space indent, `<-` assignment, snake_case).
- Roxygen with markdown (`Roxygen: list(markdown = TRUE)`).
- After any change to exports/roxygen: run `roxygen2::roxygenise()` and
  `testthat::test_dir("tests/testthat")`.
- Test fixtures that are fitted models (`tests/data/plot_test_fit.rds`) are produced
  with the CmdStanR backend; regenerate them if the draws representation changes.

## Verifying a change end-to-end

The unit tests exercise standata construction and small fits. For a real check,
fit a known example and confirm sensible dynamics:

```r
library(EpiEstim); data("Flu1918")
date <- as.Date("1918-01-01") + seq(0, along.with = c(NA, Flu1918$incidence))
d <- data.frame(city = "Baltimore", cases = c(NA, Flu1918$incidence), date = date)
fm <- epim(
  rt  = epirt(R(city, date) ~ rw(prior_scale = 0.1), prior_intercept = normal(log(2), 0.2)),
  obs = epiobs(cases ~ 0 + offset(rep(1, 93)), link = "identity", i2o = rep(.25, 4)),
  inf = epiinf(gen = Flu1918$si_distr),
  data = d, iter = 1500, chains = 3, seed = 12345)
summary(fm)          # check Rhat ≈ 1, adequate n_eff
plot_rt(fm); plot_obs(fm, type = "cases")
```
