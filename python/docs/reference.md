# API reference

The public API is re-exported from the top-level `epidemia` package.

Start with **[The model](#the-model)**. `build_epidemia_model` is the current
builder and expresses what R's `epirt`/`epiinf`/`epiobs`/`epim` do. The two older
builders, under [Superseded](#superseded), predate it and are kept only because
the example notebooks still use them.

---

## The model

::: epidemia.core.build_epidemia_model

::: epidemia.core.fit_epidemia

### Configuration

::: epidemia.core.EpiModelConfig

::: epidemia.core.ObsModel

::: epidemia.core.RandomWalk

::: epidemia.core.PanelData

### Preparing data

::: epidemia.core.prepare_panel

## Formulas

Write the model as R does, rather than assembling design matrices by hand.

::: epidemia.formula.parse_formula

::: epidemia.formula.build_from_formula

## Priors

Prior specifications mirroring R's constructors, with R's argument names. Pass
them to `EpiModelConfig(prior_covariates=, prior_intercept=, prior_seeds=,
prior_aux=)` or `ObsModel(prior_intercept=, prior=, prior_aux=)`; leaving a field
unset keeps the scalar-hyperparameter default, so nothing changes unless you ask.

::: epidemia.priors.normal

::: epidemia.priors.student_t

::: epidemia.priors.cauchy

::: epidemia.priors.exponential

::: epidemia.priors.laplace

::: epidemia.priors.shifted_gamma

::: epidemia.priors.hexp

::: epidemia.priors.decov

::: epidemia.priors.lkj

::: epidemia.priors.hs

::: epidemia.priors.hs_plus

::: epidemia.priors.lasso

::: epidemia.priors.product_normal

::: epidemia.priors.autoscale

::: epidemia.priors.build_covariance

## Inference

::: epidemia.infer.fit

::: epidemia.variational.fit_variational

## Forecasting

Forward simulation over posterior draws — nothing is refitted.

::: epidemia.forecast.forecast

::: epidemia.forecast.Forecast

::: epidemia.predict.simulate

::: epidemia.predict.posterior_predict

::: epidemia.predict.expected_observations

## Diagnostics

Check the sampler before reading anything off a fit. The mirror of R's
`sampler_diagnostics()`, reporting the same quantities under the same names.

::: epidemia.diagnostics.sampler_diagnostics

::: epidemia.diagnostics.SamplerDiagnostics

## Post-processing

::: epidemia.postprocess.posterior_linpred

::: epidemia.postprocess.posterior_infectious

::: epidemia.postprocess.extract_samples

::: epidemia.postprocess.prior_summary

## Scoring

::: epidemia.scoring.evaluate_forecast

::: epidemia.scoring.crps

::: epidemia.scoring.posterior_metrics

::: epidemia.scoring.posterior_coverage

## Plots

::: epidemia.plots.plot_rt

::: epidemia.plots.plot_obs

::: epidemia.plots.plot_infections

::: epidemia.plots.spaghetti_rt

::: epidemia.plots.spaghetti_infections

::: epidemia.plots.spaghetti_obs

::: epidemia.plots.plot_infectious

::: epidemia.plots.plot_linpred

::: epidemia.plots.plot_coverage

::: epidemia.plots.plot_metrics

::: epidemia.plots.available_series

## Renewal dynamics (NumPy reference)

The reference implementation the PyMC model is checked against.

::: epidemia.renewal

## Data

::: epidemia.data

::: epidemia.data.england_new_cases

::: epidemia.data.europe_covid

---

## Superseded

Still working, because the notebooks use them. `model.build_model` has a random
walk but a single population; `multilevel.build_multilevel_model` has regions but
a deterministic `R_t` and one observation series. Neither can express the
combination, which is why `core.build_epidemia_model` exists.

::: epidemia.model.EpiConfig

::: epidemia.model.build_model

::: epidemia.multilevel.MultilevelConfig

::: epidemia.multilevel.MultilevelData

::: epidemia.multilevel.prepare_panel

::: epidemia.multilevel.build_multilevel_model

::: epidemia.multilevel.fit_multilevel

::: epidemia.multilevel.effect_table
