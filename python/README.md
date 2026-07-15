# epidemia (Python)

A Python counterpart of the R package [**epidemia**](https://mlgh-sg.com/epidemia2/):
Bayesian **semi-mechanistic** modelling of infectious diseases. Latent daily
infections are propagated by a discrete **renewal process** (a self-exciting
point process); the time-varying reproduction number `R_t` is modelled with a
random walk (and, in future, covariates), and observations are linked to
infections through a delay/ascertainment convolution.

The model is specified with **NumPyro** (JAX) and fit with a fast NUTS backend —
**nutpie** by default, with **BlackJAX** and NumPyro's own NUTS as alternatives.

## Install (with `uv`)

```bash
cd python
uv python pin 3.12            # broad wheel availability
uv sync                       # CPU JAX everywhere (the optimized default on Apple Silicon)
```

**GPU / accelerators**

- **Linux + NVIDIA GPU:** `uv sync --extra cuda` (pulls `jax[cuda12]`; JAX uses the GPU automatically).
- **Apple Silicon:** the CPU build *is* the recommended optimized path — XLA's ARM CPU backend is fast and fully correct (float64, `lax.scan`). `jax-metal` is available via `uv sync --extra metal` but is experimental and **not recommended for this model** (no float64, and it segfaults on `lax.scan`, which the renewal recursion relies on).

Check the active backend:

```python
import jax; print(jax.default_backend(), jax.devices())
```

## Quick start

```python
import numpy as np, epidemia as epi

d = epi.flu1918()
y = np.concatenate([[np.nan], d.incidence])            # 1st day explained by seeding
config = epi.EpiConfig(
    gen=d.generation, i2o=np.repeat(0.25, 4), seed_days=6,
    link="log", intercept_loc=np.log(2.0), intercept_scale=0.2, rw_prior_scale=0.1,
)
idata = epi.fit(y, config, sampler="nutpie", draws=1000, tune=1000, chains=4)

epi.plots.plot_rt(idata)          # reproduction number over time
epi.plots.plot_obs(idata, y)      # posterior predictive vs observed
epi.plots.plot_infections(idata)  # latent infections
```

Or run the packaged example:

```bash
uv run epidemia-flu --sampler nutpie --save flu
```

## Notes

- The random walk on `R_t` uses a **non-centred** parameterisation, which avoids
  the funnel geometry that produces divergence/E-BFMI warnings in centred
  formulations.
- All latent series (`Rt`, `infections`, `E_obs`) are returned in the ArviZ
  `InferenceData` posterior, so `arviz` diagnostics and the plotting helpers work
  out of the box.
