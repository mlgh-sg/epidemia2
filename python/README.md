# epidemia (Python)

A Python counterpart of the R package [**epidemia**](https://mlgh-sg.com/epidemia2/):
Bayesian **semi-mechanistic** modelling of infectious diseases. Latent daily
infections are propagated by a discrete **renewal process** (a self-exciting
point process); the time-varying reproduction number `R_t` is modelled with a
random walk (and, in future, covariates), and observations are linked to
infections through a delay/ascertainment convolution.

The model is specified with **[PyMC](https://www.pymc.io/)** and fit with
**[nutpie](https://github.com/pymc-devs/nutpie)** — a fast Rust NUTS with
Fisher-information mass-matrix adaptation — returning an ArviZ `InferenceData`.

## Install (with `uv`)

```bash
cd python
uv python pin 3.12   # broad wheel availability
uv sync              # PyMC + nutpie, compiled with the numba CPU backend
```

The default `numba` backend is the recommended optimized path on **all**
platforms, including **Apple Silicon** — it compiles the log-density to native
code with no per-call Python overhead.

**GPU (Linux + NVIDIA):**

```bash
uv sync --extra gpu                       # pulls jax[cuda12]
# then compile the log-density with the JAX backend so it runs on the GPU:
idata = epi.fit(y, config, backend="jax")
```

**Hard posteriors:** `uv sync --extra flow` enables nutpie's normalizing-flow
adaptation (`epi.fit(..., adaptation="flow")`); `adaptation="low_rank"` needs no
extra.

## Quick start

```python
import numpy as np, epidemia as epi

d = epi.flu1918()
y = np.concatenate([[np.nan], d.incidence])            # 1st day explained by seeding
config = epi.EpiConfig(
    gen=d.generation, i2o=np.repeat(0.25, 4), seed_days=6,
    link="log", family="neg_binom",
    intercept_loc=np.log(2.0), intercept_scale=0.2, rw_prior_scale=0.1,
)
idata = epi.fit(y, config, draws=1000, tune=1000, chains=4)

epi.plots.plot_rt(idata)          # reproduction number over time
epi.plots.plot_obs(idata, y)      # posterior predictive vs observed
epi.plots.plot_infections(idata)  # latent infections
```

Every plot is also written to `./figures` (or `$EPIDEMIA_FIGDIR`); pass
`save=False` to render without writing, or `save="name"` to choose the file.

For a **multi-region** (multilevel) fit, pass the panel you fitted and each
region gets its own panel — plus effect sizes as a percent reduction in
transmission:

```python
fit = epi.prepare_panel(ec.data, epi.EUROPE_COVID_NPIS, fit_until="2020-05-05")
idata = epi.fit_multilevel(fit, config)

epi.plots.plot_rt(idata, data=fit)                       # every country, faceted
epi.plots.plot_rt(idata, data=fit, group="Italy")        # just one
epi.plots.plot_percent_effects(idata, config, data=fit)  # "% reduction in R_t"
epi.effect_table(idata, config, data=fit)                # the same, as a table
```

Or run the packaged example:

```bash
uv run epidemia-flu --save flu                 # numba (default)
uv run epidemia-flu --backend jax --save flu   # JAX backend (GPU on Linux)
```

## Performance

The model is vectorised wherever the mathematics allows:

- the **random walk** on `log R_t` is a single `cumsum` (no loop);
- the **infection→observation delay** is a *time-invariant* filter and is computed
  as a **vectorised convolution** (a sum of shifted infection series);
- the **renewal recursion** is a *time-varying* linear filter (`R_t` changes each
  step and the output feeds back), so it has **no** convolution/FFT form and stays
  a sequential scan — but each step is a single BLAS dot product (the renewal
  weight). Every comparable tool (the R `epidemia` Stan model, EpiNow2,
  epinowcast) uses the same sequential loop.

## Notes

- The random walk on `R_t` uses a **non-centred** parameterisation, which avoids
  the funnel geometry that produces divergence/E-BFMI warnings in centred
  formulations.
- All latent series (`Rt`, `infections`, `E_obs`) are returned in the ArviZ
  `InferenceData` posterior, so `arviz` diagnostics and the plotting helpers work
  out of the box.
