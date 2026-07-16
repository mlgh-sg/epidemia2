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

**Every plot is written to disk by default** — into `./figures`, or
`$EPIDEMIA_FIGDIR` if that is set. Pass `save=False` to turn it off, or
`save="name"` / `save="path/to/file.png"` to choose the destination:

```python
epi.plots.plot_rt(idata, save=False)        # render only, write nothing
epi.plots.plot_rt(idata, save="rt-final")   # -> ./figures/rt-final.png
```

### Multi-region (multilevel) fits

As in R, the same three functions handle a multi-region fit — pass the
`MultilevelData` you fitted as `data=` and you get **one panel per region**:

```python
fit = epi.prepare_panel(data, epi.EUROPE_COVID_NPIS, fit_until="2020-05-05")
idata = epi.fit_multilevel(fit, config)

epi.plots.plot_rt(idata, data=fit)                  # every country, faceted
epi.plots.plot_obs(idata, data=fit)                 # deaths PPC, faceted
epi.plots.plot_rt(idata, data=fit, group="Italy")   # just one country
```

`data=` is required (not optional) for a multi-region fit, because each region's
series is left-aligned to *its own* first modelled day — column `t` is a
different calendar date in every region, and only `data` knows which. It also
supplies each region's true length, so padded days are dropped rather than drawn.

Effect sizes for a multilevel fit:

```python
epi.plots.plot_effects(idata)                       # global beta_k
epi.plots.plot_effects(idata, group="Italy")        # Italy's beta_k + b_k
epi.plots.plot_region_effects(idata, "lockdown")    # one NPI, every region
```

!!! note "The global `beta_k` is not 'the effect in region m'"
    Under partial pooling each region's effect is `beta_k + b[m, k]`. The global
    `beta_k` is the average across regions; `plot_effects(idata, group=...)` and
    `plot_region_effects` give the per-region quantity. And when covariates are
    collinear (as the COVID NPIs are), a `beta_k` interval covering zero means
    "not separately identifiable from the other covariates", not "no effect".

!!! warning "Correlated covariates: use `adaptation="low_rank"`, and check `r_hat`"
    Divergences and `r_hat` are different failures with different cures.
    `fit_multilevel` already defaults `target_accept=0.95`, which handles the
    hierarchical funnel and warns if any divergences survive. But collinear
    covariates produce a long thin *correlation ridge* that the default diagonal
    mass matrix cannot follow — the sampler does not diverge, it just fails to
    mix, and the estimates come out biased.

    ```python
    idata = epi.fit_multilevel(fit, config, tune=2000,
                               adaptation="low_rank", target_accept=0.99)
    ```

    On the 11-country example, `"diag"` gives `beta[schools]` and
    `beta[social_distancing]` an `r_hat` of 1.08–1.11 with an **ESS of 25–36 out
    of 4000** — and their estimates move materially once `"low_rank"` lets the
    sampler traverse the ridge. A clean divergence count is *not* evidence the
    ridge is absent: always read `r_hat` and `ess_bulk` from `arviz.summary`.

### Effect sizes as a percent reduction

[`effect_table`][epidemia.effect_table] converts the coefficients into the
quantity most people want — *by what percent did this measure cut transmission?*

```python
tab = epi.effect_table(idata, config, data=fit)
epi.plots.plot_percent_effects(idata, config, data=fit)
```

It returns one row per region × term: `kind="pct"` rows are percent reductions in
`R_t`, `kind="R"` rows are the `R_0` (no measures) and `R_t` (all measures)
reproduction numbers the percentage is derived from.

!!! warning "`1 - exp(beta)` is the wrong conversion here"
    That is the answer for a **log** link. This model uses `scaled_logit(K)`, so
    `R = K * sigmoid(eta)` and a coefficient does *not* imply a constant
    multiplicative effect — the same `beta` buys a larger percentage where `R_0`
    is low than where it is high. On the Europe data the shortcut overstates
    lockdown by ~9 percentage points. `effect_table` does the counterfactual per
    posterior draw instead, which is also why it reports per region rather than
    one global figure.

Pass `data=` so each row is flagged with `enacted`: where a region never used a
measure (Sweden and lockdown), the percentage is a counterfactual extrapolation
from the pooled prior — that region's data say nothing about it —
and `plot_percent_effects` greys it out.

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
