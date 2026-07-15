## epidemia 1.1.0

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
