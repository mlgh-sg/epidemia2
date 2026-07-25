## epidemia 1.1.0

### Bug fixes

* **Models with a single group-level term now fit.** `(1 | group)` — the
  simplest multilevel specification there is — failed with `'perm' is of wrong
  length 3 (!= 2)`. With one covariance entry, `apply()` returned a matrix
  rather than a 3-d array when rebuilding `Sigma` from `theta_L`, and the
  subsequent `aperm()` errored.
* **`prior_covariance = lkj()` is now rejected with an explanatory message.**
  It was accepted by `epirt()` but the Stan program implements only `decov`, so
  the fit failed inside CmdStan with an opaque "Unable to retrieve the metadata"
  after the model had compiled. Use `decov()`; with the `||` form of a random
  effect it reduces to independent scale priors.
* **CRPS was computed against the wrong distribution.** `crps()` pooled the
  entire `[observation, draw]` matrix into a single empirical distribution and
  scored every observation against that marginal, so each day's score depended
  on the other dates and groups in the same call. With a point-mass predictive
  it returned 4.44 where the answer is 0, and on well-calibrated draws it
  overstated the score by one to two orders of magnitude. Every observation is
  now scored against its own predictive draws. This affects `crps` values from
  `evaluate_forecast()`, `posterior_metrics()` and `plot_metrics()`;
  `mean_abs_error` and `median_abs_error` were unaffected.
* **Forecast evaluation scored the wrong observation series.** In
  `evaluate_forecast()` the index was `which(type %in% alltypes)`, which is
  `which()` of a length-1 logical and so evaluated to 1 for any modelled type.
  On a model with several series, every metric and coverage number for a
  non-first `type` compared that series' predictions against the *first*
  series' observations. Affects `posterior_metrics()`, `posterior_coverage()`,
  `plot_metrics()` and `plot_coverage()` equally.
* **`posterior_predict()` returned `NULL` for multi-series models.** `types`
  defaults to every modelled type, and the return was `out[[types]]` — recursive
  indexing once `types` has length greater than one. The documented default call
  therefore returned `NULL` on any model with two or more observation series. A
  single requested type still unwraps, as `plot_obs()` expects.
* **Rows coded `-1` are no longer scored as observations.** `epiobs()` documents
  `-1` as the placeholder for a forecast horizon, and the multiple-observations
  tutorial builds `newdata` that way, but the scoring path treated them as counts
  of −1: error inflated and interval coverage collapsed to zero on exactly the
  rows being forecast. The plotting path already dropped them, so the two
  disagreed about the same `newdata`.
* **`posterior_predict(draws = n)` paired mismatched draws.** The predictive
  means were a random subsample of the posterior while the observation dispersion
  parameters were taken from the full posterior in its original order, so each
  simulated draw combined a mean from one posterior draw with an auxiliary
  parameter from another.
* `gr_subset()` no longer collapses the draws matrix to a vector when a subset
  leaves a single observation, and `plot_coverage()` no longer mis-assigns
  interval shading when `levels` is passed out of order.
* **`evaluate_forecast(groups = ...)` works with `newdata`.** The predictions
  were restricted to the requested groups but the observations were not, so the
  call failed with "replacement has N rows, data has M".
* **`plot_metrics()` now builds.** It, and the `by_group` / `by_unseen`
  branches of `plot_coverage()`, used `facet_wrap(.data$col)`, which errors with
  "Can't subset `.data` outside of a data mask context".
* **`posterior_predict()` reports which observation types exist** when given an
  unknown one; the message previously ended in a stray `FALSE`, because
  `call. = FALSE` had been placed inside `paste0()` rather than `stop()`.
* Generated quantities are now written to a directory owned by the call and read
  in a single pass, rather than lazily, once per latent series, from the shared
  session temporary directory. This is faster (one CSV parse instead of five)
  and fixes intermittent failures in sessions that fit many models — either
  "File does not exist: ...epidemia_pp_base-<stamp>.csv" or a malformed read.
* The `epim()` example no longer uses variational Bayes. ADVI could not compute
  an initial ELBO for that particular model, so the documented call failed
  outright; it now uses `algorithm = "sampling"`, with a note on when variational
  Bayes is and is not appropriate.

### Reproducibility and testing

* The R environment is locked with **renv** (`renv.lock`) and CmdStan is pinned
  to 2.36.0. `make setup` restores both; see `AGENTS.md`.
* A **`R-CMD-check` workflow** now runs `R CMD check` and the test suite on
  Ubuntu and macOS, on every branch. Previously no workflow ran the tests.
* New tests cover fitting models with **multiple observation series** and with
  **multilevel / partially pooled** effects, neither of which was previously
  fitted anywhere in the suite. Forecast evaluation is covered again.
* The test suite no longer requires `rstanarm`, which was an undeclared
  dependency of six test files even though the package itself had dropped it.

### Modernised Stan toolchain (CmdStanR)

* **Backend switched from `rstan` to `cmdstanr`.** Models are now fit with
  [CmdStanR](https://mc-stan.org/cmdstanr/) and the latest CmdStan. This is
  faster, tracks upstream Stan, and removes the fragile install-time C++
  compilation. The Stan programs are compiled on first use and cached (see
  `compile_epidemia()`).
* **Stan programs updated to modern Stan syntax** (Stan 2.33+/CmdStan 2.36),
  including the new array declaration syntax (`array[n] int x`) and the
  vertical-bar CDF calls (`normal_cdf(x | mu, sigma)`). The external C++
  `csr_matrix_times_vector2` helper is now implemented in pure Stan.
* **Posterior draws are represented with the [`posterior`](https://mc-stan.org/posterior/)
  package** rather than an `rstan` `stanfit` object. `as.matrix()`,
  `as.array()`, `as.data.frame()`, `summary()`, `posterior_rt()`,
  `posterior_predict()` etc. behave as before.
* **`rstanarm` is no longer a dependency.** The prior constructors (`normal()`,
  `student_t()`, `cauchy()`, `exponential()`, `laplace()`, `lasso()`, `hs()`,
  `hs_plus()`, `product_normal()`, `lkj()`, `decov()`) are now provided directly
  by `epidemia`, with identical behaviour. Existing code using these functions
  continues to work unchanged.
* Speed: chains run in parallel by default (`parallel_chains`), and models are
  compiled once and cached.
* No install-time compilation: the `src/`, `configure` and `LinkingTo`
  machinery has been removed.

## epidemia 1.0.0
* First version submitted to CRAN
* Bug fixed for latent infections and first Rt with pop_adjust
* reorganized files to correctly attribute copyright.
* New model for adding vaccination adjustments
* latent infections switched to normal, from log-normal
* Additional vignettes and model description
* Additional noise options for infection process 
* Significantly more flexible modeling of seeding process
* Many small bug fixes

## epidemia 0.7.0
* Changes to general interface - new epiinf() function for representing infection model
* Improved package website, with better description of the model and examples in the vignettes.
* Ability to model latent infections explicitly - replacing renewal equation
* Removed 'pop' and replaced with column of susceptibles in dataframe. 
This allows susceptible population to reduce over time due to vaccinations.
* Improved error checking in epim(), with more informative messages
* scaled_logit for epiobs
* Full integration with Bayesplot package, and a plot.epimodel method which 
easily allows the user to choose different components of the model.
* Fixed bug which meant "fullrank" was actually using "meanfield"
* Ability to use random walks in the epiobs models
* Choose between identity, scaled logit, and log link for epirt
* Plot the (potentially transformed) linear predictors for both epiobs and epirt
* Additional families for epiobs - normal and lognormal
* Add summary method and printing for epimodel objects
* Plots improved, allowing step for plot_rt, and improved formatting

## epidemia 0.6.0
* Substantial changes to interface: Added epirt and epiobs objects
* Different and more flexible observation models
* Improved structure to epimodel objects
* Refactoring of main epim function
* Improved plots, including interactive plots using plotly
* Forecast evaluation using coverage and different metrics
* Ability to do an initial run fit to cumulatives within epim
* Updated tests, documentation and vignettes

## epidemia 0.5.3
* Improved model description in introduction vignette

## epidemia 0.5.2
* Passes R CMD Check with no warnings
* Updated installation instructions

## epidemia 0.5.1
* Renamed stan files to avoid errors with Rstan 2.12.
* Plotting vignette

## epidemia 0.5.0
* Random walk terms parsed separately as input to stan files. Variance parameter sampled in stan, and so can make predictions.
* pseudo-log scales for `plot_infections` and `plot_observations`
* control over date range for all plots
* option to plot smoothed Rt in `plot_rt`

## epidemia 0.4.0
* Features for counterfactual analysis and predictions

## epidemia 0.3.3
* Added vignette describing priors
* Description of collinearity issues in 'resolving problems' vignette
* Form for potential beta testers

## epidemia 0.3.2
* Citation file
* Fixes to documentation in website

## epidemia 0.3.1
* Website fixes
* Separate index for website and github

## epidemia 0.3.0
* Website and more extensive vignettes

## epidemia 0.2.0
* Initial version.
