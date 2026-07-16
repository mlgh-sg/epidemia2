# epidemia (Python)

A Python counterpart of the R package
[**epidemia**](https://mlgh-sg.com/epidemia2/): Bayesian **semi-mechanistic**
modelling of infectious diseases.

Latent daily infections are propagated by a discrete **renewal process** (a
self-exciting point process); the time-varying reproduction number `R_t` is
modelled with a random walk, and observations are linked to infections through a
delay/ascertainment convolution.

The model is specified with **[PyMC](https://www.pymc.io/)** and fit with
**[nutpie](https://github.com/pymc-devs/nutpie)** — a fast Rust NUTS with
Fisher-information mass-matrix adaptation — returning an ArviZ `InferenceData`.

## Install

Using [`uv`](https://docs.astral.sh/uv/):

```bash
cd python
uv python pin 3.12   # broad wheel availability
uv sync              # PyMC + nutpie, compiled with the numba CPU backend
```

The default `numba` backend is the recommended optimized path on **all**
platforms, including **Apple Silicon**.

!!! tip "GPU (Linux + NVIDIA)"
    `uv sync --extra gpu` pulls `jax[cuda12]`; then fit with
    `epi.fit(y, config, backend="jax")` to run the log-density on the GPU.

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

Or run the packaged example:

```bash
uv run epidemia-flu --save flu
```

## Next steps

- The [**user guide**](guide.md) walks through the model, the configuration
  options, and the performance design.
- The [**API reference**](reference.md) documents every public function and class.
