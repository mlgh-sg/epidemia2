# Priors

Every prior in the R package has a constructor with the same name here, taking
R's argument names and reporting R's `dist` string. They live in
[`epidemia.priors`](reference.md#priors) and are passed to
`EpiModelConfig(prior_covariates=, prior_intercept=, prior_seeds=, prior_aux=)`
and `ObsModel(prior_intercept=, prior=, prior_aux=)`.

A specification only *describes* a prior. It becomes a random variable when the
model is built, so the same object can be reused across models and inspected
before fitting.

```python
from epidemia.priors import normal, shifted_gamma

normal(0, 0.5).params()          # {'location': 0.0, 'scale': 0.5}
shifted_gamma(1/6, 1, 0.05)      # the default R_t covariate prior in the vignettes
```

## What a bare constructor means

R's constructors take `scale = NULL` and resolve it late: `set_prior_scale()`
substitutes `default_scale`, which `standata_reg` passes as **0.25** for
coefficients, intercepts and auxiliary parameters alike. Python has no `NULL` to
resolve, so 0.25 is the constructor default. A bare `normal()` therefore means
`N(0, 0.25)` in both languages.

## Which families go where

R rejects an out-of-set family rather than letting it through, and so does this
port. The sets are `OK_DISTS`, `OK_INT_DISTS`, `OK_AUX_DISTS` and
`OK_COV_DISTS`.

| Slot | Allowed |
|---|---|
| Covariates (`prior_covariates`, `ObsModel(prior=)`) | `normal`, `student_t`, `cauchy`, `laplace`, `shifted_gamma`, `hexp` |
| Intercepts (`prior_intercept`) | `normal`, `student_t`, `cauchy` |
| Auxiliary (`prior_aux`) | `normal`, `student_t`, `cauchy`, `exponential` |
| Covariance (`prior_covariance`) | `decov`; `lkj` also works here, which R does not support |
| Removal noise (`prior_rm_noise`) | `normal` |

## Autoscaling

R divides a coefficient's prior scale by that covariate's own scale — the
**range** for a binary covariate, the **standard deviation** otherwise — so the
prior is a statement about standardised covariates. Whether it happens is a
property of the prior, and epidemia's defaults are the opposite of rstanarm's:

- `shifted_gamma` autoscales by default. It is the default `R_t` covariate prior
  in the vignettes, so this applies more often than it looks.
- `normal`, `student_t`, `cauchy`, `laplace` and `lasso` do **not**.

The model builder applies it; `EpiModelConfig(autoscale=False)` turns it off
entirely.

## Shrinkage families

`hs`, `hs_plus`, `lasso` and `product_normal` are available for covariate
priors when there are many covariates and you expect most effects to be near
zero. They carry no `autoscale` argument, in either language.

## Covariance priors on region effects

`decov()` is R's only covariance prior: a Gamma scale split across terms by a
symmetric Dirichlet, combined with an LKJ correlation. `lkj()` is exported by R
but its Stan program has no branch for it, so `epirt()` rejects it. Here it
works and builds a genuine LKJ covariance — a place where the port is more
capable than the original rather than merely different.

See the [parity table](parity.md) for the full mapping, and
[the reference](reference.md#priors) for each constructor's arguments.
