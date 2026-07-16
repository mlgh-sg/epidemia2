# User guide

## The model

Given an observed series `y` (with `nan` for unobserved/seeding days) and an
[`EpiConfig`][epidemia.EpiConfig], [`build_model`][epidemia.build_model]
assembles a PyMC model with the following structure.

**Transmission.** The reproduction number follows a non-centred random walk on
the link scale,

```
eta_t = intercept + cumsum(rw_scale * rw_noise)[t]
R_t   = link^{-1}(eta_t)
```

with `rw_noise ~ Normal(0, 1)`, `rw_scale ~ HalfNormal(rw_prior_scale)` and
`intercept ~ Normal(intercept_loc, intercept_scale)`. The **non-centred**
parameterisation avoids the funnel geometry that produces divergences and
E-BFMI warnings in centred random walks. Links: `"log"` (default) or
`("scaled_logit", K)` for a carrying-capacity ceiling.

**Infections.** Latent daily infections follow the discrete renewal equation

```
i_t = R_t * sum_{s} gen[s-1] * i_{t-s}
```

seeded over the first `seed_days` days by `seed ~ Exponential(1 / seed_prior_mean)`.

**Observations.** Expected observations are a causal convolution of infections
with the infection-to-observation delay `i2o`,

```
E_t = sum_{k} i2o[k] * i_{t-k}
```

fed to a Poisson, negative-binomial (default; an overdispersion term absorbs
day-to-day noise), or Normal likelihood on the observed days only.

All three latent series — `Rt`, `infections`, `E_obs` — are recorded as PyMC
`Deterministic`s, so they appear in the returned posterior for diagnostics and
plotting.

## Configuration

Every knob lives on [`EpiConfig`][epidemia.EpiConfig]:

| Field | Meaning |
| --- | --- |
| `gen` | generation-interval kernel (drop the same-day entry) |
| `i2o` | infection-to-observation delay distribution |
| `seed_days` | length of the seeding window |
| `link` | `"log"` or `("scaled_logit", K)` |
| `family` | `"poisson"`, `"neg_binom"`, or `"normal"` |
| `rw_prior_scale` | scale of the `HalfNormal` prior on the RW step size |
| `intercept_loc`, `intercept_scale` | `Normal` prior for the `R_t` intercept |
| `seed_prior_mean` | mean of the `Exponential` prior on seeded infections |
| `rw_index` | map each day to a RW step (e.g. a weekly walk); daily by default |

## Fitting

[`fit`][epidemia.fit] builds the model and samples it with nutpie:

```python
idata = epi.fit(y, config, draws=1000, tune=1000, chains=4, seed=0,
                adaptation="diag", backend="numba")
```

- **`adaptation`** — `"diag"` (Fisher-information diagonal; a strong default),
  `"low_rank"`, or `"flow"` (normalizing-flow; needs `uv sync --extra flow`) for
  harder posteriors.
- **`backend`** — `"numba"` (low-overhead CPU default, incl. Apple Silicon) or
  `"jax"` (required for GPU on Linux, `uv sync --extra gpu`).

## Plotting

The [`plots`][epidemia.plots] module uses **plotnine** (a grammar of graphics)
with publication-oriented defaults:

```python
epi.plots.plot_rt(idata)               # median + credible ribbons for R_t
epi.plots.plot_infections(idata)       # latent infections
epi.plots.plot_obs(idata, observed=y)  # posterior predictive vs observed
```

Each returns a `plotnine.ggplot` you can further theme or `.save(...)`.

## Performance

The model is vectorised wherever the mathematics allows, and sequential only
where it must be:

- the **random walk** is a single `cumsum` — no loop;
- the **infection→observation delay** is a *time-invariant* filter, computed as a
  **vectorised convolution** (a sum of shifted infection series);
- the **renewal recursion** is a *time-varying* linear filter (`R_t` changes each
  step and the output feeds back), so it has **no** convolution/FFT form and stays
  a sequential `pytensor.scan` — but each step is a single BLAS dot product (the
  renewal weight). The R `epidemia` Stan model, EpiNow2 and epinowcast all use the
  same sequential loop.

The pure-NumPy reference in [`renewal`][epidemia.renewal] mirrors this recursion
allocation-free (it reads each window straight out of the infection array) and is
used for forward/prior simulation and testing.
